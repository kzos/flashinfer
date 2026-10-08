"""CPU regression tests for deferred NVFP4 output cache identity (#6077)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import flashinfer.fused_moe.core as core_mod
from flashinfer.autotuner import AutoTuner, TuningConfig, autotune
from flashinfer.fused_moe.backends.trtllm.sm100_runner import MoERunner
from flashinfer.fused_moe.shared.inputs import (
    MoeRunnerInputs,
    alloc_trtllm_moe_output,
)
from flashinfer.tllm_enums import (
    DtypeTrtllmGen,
    Fp8QuantizationType,
    RoutingInputMode,
    SfLayout,
)

OP = "flashinfer::trtllm_fp4_block_scale_moe"
H = 5120


@pytest.fixture
def no_cuda_initialization(monkeypatch):
    # Collection may have initialized CUDA already. Guard the operations in
    # this test instead of asserting a process-global precondition.
    lazy_init = MagicMock(side_effect=AssertionError("CPU metadata must not init CUDA"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lazy_init)
    yield
    lazy_init.assert_not_called()


def _runner(**overrides):
    config = dict(
        top_k=6,
        num_local_experts=384,
        dtype_act=DtypeTrtllmGen.E2m1,
        dtype_weights=DtypeTrtllmGen.E2m1,
        fp8_quantization_type=Fp8QuantizationType.NoneFp8,
        hidden_size=H,
        intermediate_size=2304,
        num_experts=384,
        use_shuffled_weight=True,
    )
    return MoERunner(SimpleNamespace(), **(config | overrides))


def _inputs(tokens, width):
    output = alloc_trtllm_moe_output(tokens, H, width != 0, torch.device("cpu"))
    if width not in (0, H):
        output = torch.empty(tokens, width, dtype=torch.bfloat16)
    return MoeRunnerInputs(
        output=output,
        routing_logits=None,
        topk_ids=torch.zeros(tokens, 6, dtype=torch.int32),
        expert_weights=torch.ones(tokens, 6, dtype=torch.float32),
        hidden_states=torch.empty(tokens, H // 2, dtype=torch.uint8),
        hidden_states_scale=torch.empty(tokens, H // 16, dtype=torch.float8_e4m3fn),
        gemm1_lora_delta=None,
        per_token_scale=None,
    )


def _config(runner, inputs, *, do_finalize=False, **options):
    options.setdefault("routing_input_mode", RoutingInputMode.UnpackedPrecomputed)
    overrides = runner._deferred_fp4_output_key_overrides(
        inputs, do_finalize=do_finalize, **options
    )
    return runner._make_tuning_config(
        inputs,
        tune_max_num_tokens=384,
        routing_input_mode=options["routing_input_mode"],
        hidden_states_scale_layout=SfLayout.layout_linear,
        cache_key_shape_overrides=overrides,
    )


def _key(tuner, runner, inputs, config):
    tensors = inputs.to_list()
    return tuner._get_cache_key(
        OP,
        runner,
        tuner._get_input_sizes(tensors),
        config,
        runner.get_cache_key_extras(tensors),
    )


@pytest.mark.parametrize("tokens", [1, 2, 8, 33, 64, 96, 192, 384])
@pytest.mark.parametrize("per_token", [False, True])
def test_deferred_reuses_full_width_memory_and_file_key(tokens, per_token):
    tuner, runner = AutoTuner(), _runner(use_per_token_scaling=per_token)
    final, deferred = _inputs(tokens, H), _inputs(tokens, 0)
    if per_token:
        final.per_token_scale = deferred.per_token_scale = torch.ones(tokens, 1)
    final_config = _config(runner, final, do_finalize=True)
    deferred_config = _config(runner, deferred)
    original_shapes = tuner._get_input_sizes(deferred.to_list())
    final_key = _key(tuner, runner, final, final_config)
    deferred_key = _key(tuner, runner, deferred, deferred_config)
    assert final_key == deferred_key
    assert final_key.file_key == deferred_key.file_key
    # A finalized call retains its pre-fix key and no override.
    assert final_config.cache_key_shape_overrides == ()
    legacy_profile = tuner._find_nearest_profile(
        tuner._get_input_sizes(final.to_list()), final_config
    )
    assert final_key.file_key == str(
        (OP, "MoERunner", legacy_profile, runner.get_cache_key_extras(final.to_list()))
    )
    assert tuner._get_input_sizes(deferred.to_list()) == original_shapes
    assert deferred.output.shape == (tokens, 0)
    assert deferred.output.untyped_storage().nbytes() == 0
    sentinel = (901, 902)  # Synthetic cache value; never executed.
    tuner.profiling_cache[final_key] = (sentinel, None)
    hit, _, tactic, _ = tuner.search_cache(
        OP, [runner], original_shapes, deferred_config, inputs=deferred.to_list()
    )
    assert hit and tactic == sentinel


def test_deferred_profile_allocations_remain_zero_width():
    tuner, runner = AutoTuner(), _runner()
    inputs = _inputs(8, 0)
    config = _config(runner, inputs)
    profiles = tuner._generate_optimization_profiles(config, inputs.to_list())
    assert profiles
    for profile in profiles:
        assert profile.get_opt_shapes()[0][1] == 0
        synthesized = tuner._prepare_input_tensors(profile, inputs.to_list())
        assert synthesized[0].shape[1] == 0
        assert synthesized[0].untyped_storage().nbytes() == 0
        key = tuner._get_cache_key(
            OP,
            runner,
            profile.get_opt_shapes(),
            config,
            runner.get_cache_key_extras(synthesized),
        )
        assert key.nearest_profile[0][1] == H


@pytest.mark.parametrize(
    "final_tokens,deferred_tokens,same_bucket",
    [(33, 64, True), (65, 96, True), (32, 33, False), (64, 65, False)],
)
def test_output_override_preserves_token_bucket_boundaries(
    final_tokens, deferred_tokens, same_bucket
):
    tuner, runner = AutoTuner(), _runner()
    final, deferred = _inputs(final_tokens, H), _inputs(deferred_tokens, 0)
    final_key = _key(tuner, runner, final, _config(runner, final, do_finalize=True))
    deferred_key = _key(tuner, runner, deferred, _config(runner, deferred))
    assert (final_key == deferred_key) is same_bucket


def test_legacy_finalized_file_cache_serves_deferred(tmp_path):
    tuner, runner = AutoTuner(), _runner()
    final, deferred = _inputs(8, H), _inputs(8, 0)
    key = _key(tuner, runner, final, _config(runner, final, do_finalize=True))
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({key.file_key: ["MoERunner", [901, 902]]}))
    assert tuner.load_configs(str(path))
    hit, _, tactic, _ = tuner.search_cache(
        OP,
        [runner],
        tuner._get_input_sizes(deferred.to_list()),
        _config(runner, deferred),
        inputs=deferred.to_list(),
    )
    assert hit and tuple(tactic) == (901, 902)


def test_warm_tuning_context_never_prepares_or_profiles_deferred(monkeypatch):
    tuner, runner = AutoTuner(), _runner()
    # Keep instance-method patches off the shared singleton so subsequent
    # class-level patches in the OOM/cleanup tests are not shadowed.
    monkeypatch.setattr(AutoTuner, "get", staticmethod(lambda: tuner))
    final, deferred = _inputs(8, H), _inputs(8, 0)
    final_config = _config(runner, final, do_finalize=True)
    deferred_config = _config(runner, deferred)
    sentinel = (901, 902)
    winners = {}
    for profile in tuner._generate_optimization_profiles(final_config, final.to_list()):
        key = tuner._get_cache_key(
            OP,
            runner,
            profile.get_opt_shapes(),
            final_config,
            runner.get_cache_key_extras(final.to_list()),
        )
        winners[key] = (sentinel, profile)
    monkeypatch.setattr(tuner, "profiling_cache", winners)

    def unexpected_cold_path(*args, **kwargs):
        pytest.fail("a warm deferred cache hit must not prepare or profile tensors")

    monkeypatch.setattr(tuner, "_prepare_input_tensors", unexpected_cold_path)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", unexpected_cold_path)
    with autotune(True):
        chosen, tactic = tuner.choose_one(
            OP, [runner], deferred_config, deferred.to_list()
        )
    assert chosen is runner and tactic == sentinel
    assert deferred.output.untyped_storage().nbytes() == 0


@pytest.mark.parametrize(
    "option,value",
    [
        ("do_finalize", True),
        ("routing_input_mode", RoutingInputMode.FromLogits),
        ("valid_hidden_size", H),
        ("valid_intermediate_size", 2304),
    ],
)
def test_excluded_call_options_keep_original_key(option, value):
    runner, inputs = _runner(), _inputs(8, 0)
    options = dict(
        do_finalize=False, routing_input_mode=RoutingInputMode.UnpackedPrecomputed
    )
    options[option] = value
    assert runner._deferred_fp4_output_key_overrides(inputs, **options) == ()


@pytest.mark.parametrize(
    "option,value",
    [
        ("dtype_act", DtypeTrtllmGen.MxE4m3),
        ("dtype_weights", DtypeTrtllmGen.MxE2m1),
        ("fp8_quantization_type", Fp8QuantizationType.MxFp8),
        ("hidden_size_output", H - 128),
        ("num_fused_shared_experts", 1),
    ],
)
def test_excluded_runner_options_keep_original_key(option, value):
    runner = _runner(**{option: value})
    assert _config(runner, _inputs(8, 0)).cache_key_shape_overrides == ()


@pytest.mark.parametrize("width", [128, H - 128, H])
def test_nonzero_output_shapes_not_canonicalized(width):
    runner = _runner()
    assert _config(runner, _inputs(8, width)).cache_key_shape_overrides == ()


@pytest.mark.parametrize("variant", ["lora", "dtype", "rank", "token_count"])
def test_unusual_outputs_and_lora_not_canonicalized(variant):
    runner, inputs = _runner(), _inputs(8, 0)
    if variant == "lora":
        inputs.gemm1_lora_delta = torch.empty(8, 6, 2304)
    elif variant == "dtype":
        inputs.output = inputs.output.to(torch.float32)
    elif variant == "rank":
        inputs.output = torch.empty(0, dtype=torch.bfloat16)
    else:
        inputs.output = torch.empty(1, 0, dtype=torch.bfloat16)
    assert (
        runner._deferred_fp4_output_key_overrides(
            inputs,
            do_finalize=False,
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        )
        == ()
    )


def test_key_only_overrides_leave_bucketing_and_other_inputs_unchanged(
    no_cuda_initialization,
):
    runner, tuner = _runner(), AutoTuner()
    raw_shapes = ((8, 0), (7, 9))
    config = TuningConfig(cache_key_shape_overrides=((0, 1, H),))
    key = tuner._get_cache_key(OP, runner, raw_shapes, config, ("identity",))
    assert key.nearest_profile == ((8, H), (7, 9))
    assert key.extras == ("identity",)
    assert raw_shapes == ((8, 0), (7, 9))
    assert tuner._find_nearest_profile(raw_shapes, config) == raw_shapes


def test_tuning_context_and_measurement_overrides_preserve_key_metadata():
    tuner, runner = AutoTuner.get(), _runner()
    config = _config(runner, _inputs(8, 0))
    with autotune(
        False, tuning_buckets=(8, 16, 64), round_up=True, cuda_graph_profile_replays=3
    ):
        overridden = tuner._apply_tuning_overrides(config)
    assert overridden.cache_key_shape_overrides == config.cache_key_shape_overrides
    measured = tuner._apply_measure_policy(
        overridden, SimpleNamespace(use_cuda_graph=True, cold_l2=True)
    )
    assert measured.cache_key_shape_overrides == config.cache_key_shape_overrides


def test_packed_precomputed_routing_is_also_admitted():
    runner = _runner()
    config = _config(
        runner, _inputs(8, 0), routing_input_mode=RoutingInputMode.PackedPrecomputed
    )
    assert config.cache_key_shape_overrides == ((0, 1, H),)


@pytest.mark.parametrize(
    "option,value",
    [
        ("top_k", 4),
        ("num_local_experts", 96),
        ("num_experts", 512),
        ("intermediate_size", 2048),
        ("activation_type", 1),
        ("use_per_token_scaling", True),
    ],
)
def test_key_overrides_preserve_runner_configuration_identity(option, value):
    tuner, inputs = AutoTuner(), _inputs(8, 0)
    runner, variant = _runner(), _runner(**{option: value})
    base_key = _key(tuner, runner, inputs, _config(runner, inputs))
    changed_key = _key(tuner, variant, inputs, _config(variant, inputs))
    assert base_key.nearest_profile == changed_key.nearest_profile
    assert base_key.extras != changed_key.extras
    assert base_key.file_key != changed_key.file_key


@pytest.mark.parametrize("tokens", [1, 2, 8, 33, 64, 96, 192, 384])
@pytest.mark.parametrize("packed", [False, True])
def test_public_wrapper_reuses_finalized_key_without_output_payload(
    monkeypatch, no_cuda_initialization, tokens, packed
):
    # Native loading/execution and tactic timing are intentionally mocked: this
    # checks the real Python wrapper, allocation and cache wiring, not numerics.
    native = MagicMock()
    native.trtllm_fp4_block_scale_moe.return_value = [
        torch.empty(0, H, dtype=torch.bfloat16),
        None,
        torch.empty(tokens, 6, dtype=torch.int32),
    ]
    module = SimpleNamespace(
        build_and_load=lambda: native, get_library_paths=lambda: []
    )
    monkeypatch.setattr(
        core_mod, "gen_trtllm_gen_fused_moe_sm100_module", lambda **_: module
    )
    runtime = core_mod._get_trtllm_moe_sm100_module_impl.__wrapped__(False)
    monkeypatch.setattr(core_mod, "get_trtllm_moe_sm100_module", lambda: runtime)
    monkeypatch.setattr(core_mod, "_device_support_moe_pdl", lambda _: False)
    monkeypatch.setattr(core_mod, "_enabled_trtllm_da_config", lambda: None)

    tuner, calls = AutoTuner(), []
    sentinel = (901, 902)

    def choose_one(op, runners, config, inputs, **kwargs):
        shapes = tuner._get_input_sizes(inputs)
        runner = runners[0]
        assert runner.dtype_act == runner.dtype_weights == DtypeTrtllmGen.E2m1
        key = tuner._get_cache_key(
            op, runner, shapes, config, runner.get_cache_key_extras(inputs)
        )
        hit, _, tactic, _ = tuner.search_cache(
            op, runners, shapes, config, inputs=inputs
        )
        calls.append((hit, key, inputs[0], kwargs["do_finalize"]))
        if not hit:
            tuner.profiling_cache[key] = (sentinel, None)
        # Exercise choose_one's real warm/non-tuning path after synthetic
        # population; only native timing/population is replaced by the fixture.
        return AutoTuner.choose_one(tuner, op, runners, config, inputs, **kwargs)

    monkeypatch.setattr(tuner, "choose_one", choose_one)
    monkeypatch.setattr(AutoTuner, "get", staticmethod(lambda: tuner))
    inputs = _inputs(tokens, 0)
    routes = (
        inputs.topk_ids << 16 if packed else (inputs.topk_ids, inputs.expert_weights)
    )
    kwargs = dict(
        topk_ids=routes,
        routing_bias=None,
        hidden_states=inputs.hidden_states,
        hidden_states_scale=inputs.hidden_states_scale,
        gemm1_weights=torch.empty(1, 1, 8, dtype=torch.uint8),
        gemm1_weights_scale=torch.empty(1, dtype=torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=torch.empty(1, 1, 8, dtype=torch.uint8),
        gemm2_weights_scale=torch.empty(1, dtype=torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=384,
        top_k=6,
        n_group=None,
        topk_group=None,
        intermediate_size=2304,
        local_expert_offset=0,
        local_num_experts=384,
        routed_scaling_factor=None,
        enable_pdl=False,
        tune_max_num_tokens=384,
        hidden_states_scale_layout=SfLayout.layout_linear,
    )
    finalized = core_mod.trtllm_fp4_block_scale_routed_moe(**kwargs, do_finalize=True)
    deferred = core_mod.trtllm_fp4_block_scale_routed_moe(**kwargs, do_finalize=False)
    assert len(finalized) == 1 and len(deferred) == 3
    assert [call[0] for call in calls] == [False, True]
    assert calls[0][1] == calls[1][1]
    assert calls[0][2].shape == (tokens, H)
    assert calls[1][2].shape == (tokens, 0)
    assert calls[1][2].untyped_storage().nbytes() == 0
    assert [call[3] for call in calls] == [True, False]
    native_args = native.trtllm_fp4_block_scale_moe.call_args.args
    assert native_args[31] is False
    assert native_args[34] is calls[1][2]
    assert native_args[35] == sentinel
