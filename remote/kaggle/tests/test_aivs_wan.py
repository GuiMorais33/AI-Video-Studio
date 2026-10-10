"""Testes do notebook do Kaggle na CPU, sem GPU e sem baixar modelos.

O fluxo completo (main) roda com:
- um pacote de verdade gerado pelo motor do estúdio;
- um "repositório" minúsculo no formato Wan-AI/Wan2.2-Animate-14B-Diffusers, salvo em disco;
- o transformer minúsculo convertido para GGUF no formato original do Wan (como os da
  QuantStack), com o codificador de movimento quantizado;
- LoRAs minúsculas no formato da Kijai (uma válida e uma com chaves desconhecidas);
- o ViTPose oficial do Wan 2.2 (onnxruntime) com um ONNX minúsculo no mesmo formato de pasta.

Requer: diffusers==0.41.0 transformers peft accelerate gguf opencv-python-headless
matplotlib onnx onnxruntime e o clone do Wan2.2 (variável WAN_DIR ou download pelo git).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("diffusers")
pytest.importorskip("gguf")
cv2 = pytest.importorskip("cv2")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aivs_wan  # noqa: E402

WAN_COMMIT = aivs_wan.Config.wan_commit


# --------------------------------------------------------------------------- unidades

def test_dilate_matches_opencv():
    rng = np.random.default_rng(0)
    mask = rng.random((60, 80)) > 0.97
    ours = aivs_wan.dilate(mask, 7, 3)
    ref = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=3) > 0
    assert np.array_equal(ours, ref)


def test_blockify_fills_touched_blocks():
    mask = np.zeros((48, 64), bool)
    mask[17, 33] = True
    out = aivs_wan.blockify(mask, 16)
    assert out[16:32, 32:48].all() and out.sum() == 256
    odd = aivs_wan.blockify(np.ones((20, 20), bool), 16)
    assert odd.shape == (20, 20) and odd.all()


@pytest.mark.parametrize(("frames", "maximum", "expected"), [(16, 77, 17), (17, 77, 17), (48, 77, 49), (150, 77, 77), (5, 77, 5)])
def test_segment_length(frames, maximum, expected):
    seg = aivs_wan.segment_length(frames, maximum)
    assert seg == expected and (seg - 1) % 4 == 0


def test_backgrounds_black_out_generated_area():
    frame = np.full((32, 32, 3), 200, np.uint8)
    mask = np.zeros((32, 32), bool)
    mask[8:24, 8:24] = True
    bg = np.asarray(aivs_wan.backgrounds([frame], [mask])[0])
    assert (bg[mask] == 0).all() and (bg[~mask] == 200).all()


def test_mask_bbox():
    mask = np.zeros((40, 50), bool)
    mask[10:20, 5:15] = True
    assert aivs_wan.mask_bbox(mask).tolist() == [5, 10, 15, 20, 1.0]
    assert aivs_wan.mask_bbox(np.zeros((4, 4), bool)) is None


# --------------------------------------------------------------------------- modelos minúsculos

@pytest.fixture(scope="session")
def wan_dir(tmp_path_factory) -> Path:
    env = os.environ.get("WAN_DIR")
    if env and Path(env, "wan", "modules", "animate", "preprocess").exists():
        return Path(env)
    root = tmp_path_factory.mktemp("wan")
    try:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "fetch", "-q", "--depth", "1",
                        "https://github.com/Wan-Video/Wan2.2.git", WAN_COMMIT], check=True, timeout=300)
        subprocess.run(["git", "-C", str(root), "checkout", "-q", "FETCH_HEAD"], check=True)
    except Exception as exc:  # sem rede
        pytest.skip(f"repositório Wan2.2 indisponível: {exc}")
    return root


TRANSFORMER = dict(
    patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=32, in_channels=36, latent_channels=16,
    out_channels=16, text_dim=64, freq_dim=256, ffn_dim=128, num_layers=3, cross_attn_norm=True,
    qk_norm="rms_norm_across_heads", image_dim=32, added_kv_proj_dim=64, rope_max_seq_len=64,
    motion_encoder_channel_sizes={"4": 32, "8": 32, "16": 32}, motion_encoder_size=16, motion_style_dim=32,
    motion_dim=4, motion_encoder_dim=32, face_encoder_hidden_dim=32, face_encoder_num_heads=2,
    inject_face_latents_blocks=3,
)


def _to_original(sd: dict) -> dict:
    """Nomes do diffusers -> nomes originais do Wan (inverso de convert_wan_transformer_to_diffusers)."""
    out = {}
    n_res = len({k.split(".")[2] for k in sd if k.startswith("motion_encoder.res_blocks.")})
    for k, v in sd.items():
        o = k
        if o.startswith("motion_encoder."):
            rep = {
                "motion_encoder.motion_synthesis_weight": "motion_encoder.dec.direction.weight",
                "motion_encoder.conv_in.weight": "motion_encoder.enc.net_app.convs.0.0.weight",
                "motion_encoder.conv_in.act_fn.bias": "motion_encoder.enc.net_app.convs.0.1.bias",
                "motion_encoder.conv_out.weight": "motion_encoder.enc.net_app.convs.8.weight",
            }
            for i in range(n_res):
                c = i + 1
                rep.update({
                    f"motion_encoder.res_blocks.{i}.conv1.weight": f"motion_encoder.enc.net_app.convs.{c}.conv1.0.weight",
                    f"motion_encoder.res_blocks.{i}.conv1.act_fn.bias": f"motion_encoder.enc.net_app.convs.{c}.conv1.1.bias",
                    f"motion_encoder.res_blocks.{i}.conv2.weight": f"motion_encoder.enc.net_app.convs.{c}.conv2.1.weight",
                    f"motion_encoder.res_blocks.{i}.conv2.act_fn.bias": f"motion_encoder.enc.net_app.convs.{c}.conv2.2.bias",
                    f"motion_encoder.res_blocks.{i}.conv_skip.weight": f"motion_encoder.enc.net_app.convs.{c}.skip.1.weight",
                })
            o = rep.get(o, o.replace("motion_encoder.motion_network", "motion_encoder.enc.fc"))
            if ".enc.net_app.convs." in o and o.endswith(".bias"):
                v = v.reshape(1, -1, 1, 1)
        elif o.startswith("face_encoder."):
            for a in ("conv1_local", "conv2", "conv3"):
                o = o.replace(f"face_encoder.{a}.", f"face_encoder.{a}.conv.")
        elif o.startswith("face_adapter."):
            o = o.replace("face_adapter.", "face_adapter.fuser_blocks.", 1)
            o = o.replace(".norm_q.", ".q_norm.").replace(".norm_k.", ".k_norm.")
            o = o.replace(".to_q.", ".linear1_q.").replace(".to_out.", ".linear2.")
            if ".to_k." in o:
                out[o.replace(".to_k.", ".linear1_kv.")] = torch.cat([v, sd[k.replace(".to_k.", ".to_v.")]], 0)
                continue
            if ".to_v." in o:
                continue
        else:
            o = (o.replace("condition_embedder.time_embedder.linear_1", "time_embedding.0")
                  .replace("condition_embedder.time_embedder.linear_2", "time_embedding.2")
                  .replace("condition_embedder.text_embedder.linear_1", "text_embedding.0")
                  .replace("condition_embedder.text_embedder.linear_2", "text_embedding.2")
                  .replace("condition_embedder.time_proj", "time_projection.1")
                  .replace("condition_embedder.image_embedder.norm1", "img_emb.proj.0")
                  .replace("condition_embedder.image_embedder.ff.net.0.proj", "img_emb.proj.1")
                  .replace("condition_embedder.image_embedder.ff.net.2", "img_emb.proj.3")
                  .replace("condition_embedder.image_embedder.norm2", "img_emb.proj.4")
                  .replace(".attn1.", ".self_attn.").replace(".attn2.", ".cross_attn.")
                  .replace(".to_out.0.", ".o.").replace(".to_q.", ".q.").replace(".to_k.", ".k.")
                  .replace(".to_v.", ".v.").replace(".add_k_proj.", ".k_img.").replace(".add_v_proj.", ".v_img.")
                  .replace(".norm_added_k.", ".norm_k_img.")
                  .replace("ffn.net.0.proj", "ffn.0").replace("ffn.net.2", "ffn.2"))
            if o.startswith("blocks.") and ".norm2." in o:
                o = o.replace(".norm2.", ".norm3.")
            if o.startswith("blocks.") and o.endswith(".scale_shift_table"):
                o = o.replace(".scale_shift_table", ".modulation")
            if o == "scale_shift_table":
                o = "head.modulation"
            o = o.replace("proj_out.", "head.head.")
        out[o] = v
    return out


def _write_gguf(orig: dict, path: Path) -> int:
    """Quantiza em Q8_0 tudo que as regras do ComfyUI-GGUF quantizariam, inclusive motion_encoder."""
    import gguf

    skip = ("modulation", "patch_embedding.", "text_embedding.", "time_projection.", "time_embedding.", "img_emb.", "head.")
    writer = gguf.GGUFWriter(str(path), arch="wan")
    quantized = 0
    for name, tensor in orig.items():
        array = tensor.detach().float().numpy()
        if array.ndim == 2 and array.shape[-1] % 32 == 0 and not any(s in name for s in skip):
            writer.add_tensor(name, gguf.quants.quantize(array, gguf.GGMLQuantizationType.Q8_0),
                              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            quantized += 1
        else:
            writer.add_tensor(name, array.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return quantized


def _kijai_lora(blocks: int, inner: int, rank: int = 4, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for i in range(blocks):
        for n in [f"self_attn.{o}" for o in "qkvo"] + [f"cross_attn.{o}" for o in ("q", "k", "v", "o", "k_img", "v_img")]:
            sd[f"diffusion_model.blocks.{i}.{n}.lora_down.weight"] = torch.randn(rank, inner, generator=g) * 0.01
            sd[f"diffusion_model.blocks.{i}.{n}.lora_up.weight"] = torch.randn(inner, rank, generator=g) * 0.01
            sd[f"diffusion_model.blocks.{i}.{n}.alpha"] = torch.tensor(float(rank))
    return sd


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory) -> dict:
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, PreTrainedTokenizerFast,
                              UMT5Config, UMT5EncoderModel)

    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanAnimatePipeline, WanAnimateTransformer3DModel

    root = tmp_path_factory.mktemp("modelos")
    words = "<pad> </s> <unk> people in the video are doing actions a robot dancing".split()
    tok = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    torch.manual_seed(0)
    text_encoder = UMT5EncoderModel(UMT5Config(vocab_size=len(words), d_model=64, d_kv=8, d_ff=64, num_layers=2,
                                               num_heads=2, relative_attention_num_buckets=8)).eval()
    vae = AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True])
    transformer = WanAnimateTransformer3DModel(**TRANSFORMER).eval()
    image_encoder = CLIPVisionModel(CLIPVisionConfig(hidden_size=32, projection_dim=32, num_hidden_layers=2,
                                                     num_attention_heads=2, image_size=4, intermediate_size=16,
                                                     patch_size=1))
    repo = root / "repo"
    WanAnimatePipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=FlowMatchEulerDiscreteScheduler(shift=8.0),
        image_processor=CLIPImageProcessor(crop_size=4, size=4), image_encoder=image_encoder, transformer=transformer,
    ).save_pretrained(repo)
    gguf_path = root / "tiny.gguf"
    quantized = _write_gguf(_to_original({k: v.clone() for k, v in transformer.state_dict().items()}), gguf_path)
    assert any(True for _ in [quantized])
    inner = TRANSFORMER["num_attention_heads"] * TRANSFORMER["attention_head_dim"]
    lightx2v = root / "lightx2v.safetensors"
    save_file(_kijai_lora(TRANSFORMER["num_layers"], inner, seed=1), str(lightx2v))
    relight = root / "relight.safetensors"
    save_file({"diffusion_model.unknown_module.lora_down.weight": torch.zeros(4, 8),
               "diffusion_model.unknown_module.lora_up.weight": torch.zeros(8, 4)}, str(relight))
    return {"repo": repo, "gguf": gguf_path, "lightx2v": lightx2v, "relight": relight}


@pytest.fixture(scope="session")
def tiny_vitpose(tmp_path_factory) -> Path:
    """ONNX minúsculo no MESMO formato do oficial: pasta vitpose_h_wholebody.onnx/ com end2end.onnx
    e os pesos em arquivos externos. Entrada (1,3,256,192) -> heatmaps (1,133,64,48)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(rng.normal(size=(133, 3, 1, 1)).astype(np.float32), "head.weight")
    bias = numpy_helper.from_array(rng.normal(size=(133,)).astype(np.float32), "head.bias")
    graph = helper.make_graph(
        [helper.make_node("AveragePool", ["input"], ["pooled"], kernel_shape=[4, 4], strides=[4, 4]),
         helper.make_node("Conv", ["pooled", "head.weight", "head.bias"], ["heatmaps"])],
        "vitpose_tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 256, 192])],
        [helper.make_tensor_value_info("heatmaps", TensorProto.FLOAT, [1, 133, 64, 48])],
        initializer=[weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    folder = tmp_path_factory.mktemp("pose2d") / "vitpose_h_wholebody.onnx"
    folder.mkdir()
    onnx.save_model(model, str(folder / "end2end.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=False, size_threshold=0)
    assert len(list(folder.iterdir())) == 3  # end2end.onnx + 2 tensores externos
    return folder


# --------------------------------------------------------------------------- pacote real do motor

@pytest.fixture(scope="session")
def package(tmp_path_factory) -> Path:
    pytest.importorskip("studio")
    from fastapi.testclient import TestClient

    from studio.api import create_app
    from studio.config import Settings

    tmp = tmp_path_factory.mktemp("estudio")
    video = tmp / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=16:duration=1",
                    "-f", "lavfi", "-i", "sine=duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(video)], check=True)
    ref = tmp / "ref.png"
    Image.new("RGB", (384, 640), (30, 90, 200)).save(ref)
    with TestClient(create_app(Settings(data_dir=tmp / "data", tracker="demo"))) as client:
        with open(video, "rb") as f:
            pid = client.post("/projects", files={"file": ("v.mp4", f)}).json()["id"]
        client.post(f"/projects/{pid}/clicks", json={"frame_idx": 0, "x": 320, "y": 180, "label": 1})
        client.post(f"/projects/{pid}/track")
        client.app.state.jobs.wait_idle(timeout=60)
        with open(ref, "rb") as f:
            client.post(f"/projects/{pid}/reference", files={"file": ("r.png", f)})
        resp = client.post(f"/projects/{pid}/wan-package", json={"resolution": "480p", "prompt": "a robot dancing"})
        client.app.state.jobs.wait_idle(timeout=120)
        job = client.get(f"/jobs/{resp.json()['id']}").json()
        assert job["status"] == "done", job
    return tmp / "data" / "projects" / pid / "exports" / "wan_package.zip"


# --------------------------------------------------------------------------- FaceFusion falso

FAKE_FACEFUSION = r'''
"""FaceFusion falso: guarda os argumentos e devolve o vídeo com as cores invertidas."""
import json, subprocess, sys
from pathlib import Path

args = sys.argv[1:]
here = Path(__file__).parent
(here / "args.json").write_text(json.dumps(args))
mode = (here / "mode").read_text().strip() if (here / "mode").exists() else "ok"
value = lambda flag: args[args.index(flag) + 1]
sources = args[args.index("-s") + 1:args.index("-t")]
if mode == "noface":
    print("[FACEFUSION.FACE_SWAPPER] no source face detected", flush=True)
    sys.exit(1)
assert all(Path(s).exists() for s in sources), sources
sys.stdout.write("analysing: 100%|##| 16/16\rprocessing:  50%|#  | 8/16\rprocessing: 100%|##| 16/16\n")
sys.stdout.flush()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", value("-t"), "-vf", "negate", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", value("-o")], check=True)
'''


@pytest.fixture
def fake_facefusion(tmp_path) -> Path:
    root = tmp_path / "facefusion"
    root.mkdir()
    (root / "facefusion.py").write_text(FAKE_FACEFUSION)
    return root


def _config(tmp_path: Path, tiny: dict, wan_dir: Path, input_root: Path, pose: Path, **kw) -> aivs_wan.Config:
    kw.setdefault("facefusion_dir", str(tmp_path / "facefusion"))
    return aivs_wan.Config(
        repo=str(tiny["repo"]), gguf_file=str(tiny["gguf"]),
        loras=((str(tiny["lightx2v"]), "lightx2v", 1.0, True), (str(tiny["relight"]), "relight", 1.0, False)),
        wan_dir=str(wan_dir), pose_path=str(pose), steps=2, dtype="float32", input_root=str(input_root),
        scratch=str(tmp_path / "scratch"), output=str(tmp_path / "working"), **kw,
    )


def _mean_luma(path: Path) -> float:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], check=True, capture_output=True).stdout
    return float(np.frombuffer(raw, np.uint8).mean())


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]


@pytest.mark.parametrize("unzipped", [False, True], ids=["zip", "descompactado"])
def test_main_end_to_end_on_cpu(tmp_path, tiny_models, tiny_vitpose, wan_dir, package, unzipped, capsys,
                                fake_facefusion):
    input_root = tmp_path / "input" / "aivs-teste"
    input_root.mkdir(parents=True)
    if unzipped:  # o Kaggle pode descompactar o .zip ao criar o dataset
        with zipfile.ZipFile(package) as zf:
            zf.extractall(input_root / "wan_package")
    else:
        (input_root / "wan_package.zip").write_bytes(package.read_bytes())
    (input_root / "aivs_run.json").write_text(json.dumps({"run_id": "r123", "project_id": "p"}))
    cfg = _config(tmp_path, tiny_models, wan_dir, tmp_path / "input", tiny_vitpose)
    report = aivs_wan.main(cfg)

    out = Path(cfg.output)
    assert report["ok"] is True
    saved = json.loads((out / "relatorio.json").read_text())
    assert saved["ok"] is True and saved["loras"]["lightx2v"] > 0
    assert any("relight" in a for a in saved["avisos"])  # LoRA opcional inválida vira aviso
    assert saved["geracao"]["segment_frames"] == 17 and saved["geracao"]["segments"] == 1
    job = saved["job"]
    video = _probe(out / "resultado.mp4")
    assert (video["width"], video["height"]) == (job["width"], job["height"]) == (832, 464)
    assert video["avg_frame_rate"] == job["fps"] == "16/1"
    assert int(video["nb_read_frames"]) == job["num_frames"] == 16
    for name in ("pose.mp4", "rosto.mp4", "mascara.mp4"):
        assert (out / "depuracao" / name).exists()
    assert _probe(out / "depuracao" / "rosto.mp4")["width"] == 512
    stdout = capsys.readouterr().out.splitlines()
    assert report["run_id"] == "r123" and stdout.index("AIVS_RUN r123") < min(
        i for i, l in enumerate(stdout) if l.startswith("AIVS_PROGRESS"))  # o estúdio ignora progresso antes disso
    lines = [l for l in stdout if l.startswith("AIVS_PROGRESS")]
    values = [float(l.split()[1]) for l in lines]
    assert values == sorted(values) and values[-1] == 1.0
    summary = json.loads(next(l for l in stdout if l.startswith("AIVS_RELATORIO ")).removeprefix("AIVS_RELATORIO "))
    assert summary["ok"] is True and summary["run_id"] == "r123" and "traceback" not in summary

    # Refino do rosto (FaceFusion falso): o resultado é o vídeo refinado; o do Wan fica na depuração.
    assert saved["rosto"]["ok"] is True and saved["rosto"]["fotos"] == 1
    assert "refino_rosto_s" in saved["etapas"]
    ff_args = json.loads((fake_facefusion / "args.json").read_text())
    assert ff_args[ff_args.index("-s") + 1].endswith("reference.png")
    assert ff_args[ff_args.index("--face-selector-mode") + 1] == "one"
    plain = out / "depuracao" / "wan_sem_refino.mp4"
    assert abs(_mean_luma(out / "resultado.mp4") - (255 - _mean_luma(plain))) < 8  # cores invertidas
    assert any("Refinando o rosto: 100%" in l for l in lines)


def test_main_reports_errors(tmp_path, tiny_models, tiny_vitpose, wan_dir, capsys):
    empty = tmp_path / "input"
    empty.mkdir()
    cfg = _config(tmp_path, tiny_models, wan_dir, empty, tiny_vitpose)
    with pytest.raises(FileNotFoundError):
        aivs_wan.main(cfg)
    saved = json.loads((Path(cfg.output) / "relatorio.json").read_text())
    assert saved["ok"] is False and "wan_package.zip" in saved["erro"]
    error_lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("AIVS_ERRO ")]
    assert len(error_lines) == 1 and "wan_package.zip" in error_lines[0]


def test_encode_prompt_keeps_no_graph(tiny_models):
    """Sem grafo de gradiente: senão o embedding prenderia o codificador de texto na GPU."""
    cfg = aivs_wan.Config(repo=str(tiny_models["repo"]), dtype="float32")
    embeds = aivs_wan.encode_prompt(cfg, "a robot dancing", "cpu")
    assert embeds.grad_fn is None and not embeds.requires_grad and embeds.device.type == "cpu"


def test_pose_device_prefers_second_gpu(monkeypatch):
    torch = aivs_wan._torch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert aivs_wan.pose_device() == "cuda:1"
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert aivs_wan.pose_device() == "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert aivs_wan.pose_device() == "cpu"


# --------------------------------------------------------------------------- refino do rosto

def _clip(path: Path) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=16:duration=0.5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_face_sources_prefers_face_photos(tmp_path):
    (tmp_path / "reference.png").write_bytes(b"x")
    assert aivs_wan.face_sources(tmp_path) == [tmp_path / "reference.png"]
    faces = tmp_path / "faces"
    faces.mkdir()
    for name in ("02.jpg", "01.png", "nota.txt"):
        (faces / name).write_bytes(b"x")
    assert aivs_wan.face_sources(tmp_path) == [faces / "01.png", faces / "02.jpg"]


def test_facefusion_command_uses_every_photo(tmp_path):
    cfg = aivs_wan.Config()
    photos = [tmp_path / "a.png", tmp_path / "b.png"]
    cmd = aivs_wan.facefusion_command(cfg, photos, tmp_path / "in.mp4", tmp_path / "out.mp4", tmp_path, "cuda", 1)
    assert cmd[cmd.index("-s") + 1:cmd.index("-t")] == [str(p) for p in photos]
    assert cmd[cmd.index("--processors") + 1:cmd.index("--processors") + 3] == ["face_swapper", "face_enhancer"]
    assert cmd[cmd.index("--execution-providers") + 1] == "cuda" and cmd[cmd.index("--execution-device-ids") + 1] == "1"
    assert cmd[cmd.index("--face-swapper-model") + 1] == "hyperswap_1a_256"


def test_refine_face_falls_back_when_photo_has_no_face(tmp_path, fake_facefusion):
    (fake_facefusion / "mode").write_text("noface")
    cfg = aivs_wan.Config(facefusion_dir=str(fake_facefusion), output=str(tmp_path / "working"))
    report = aivs_wan.Report(tmp_path / "working" / "relatorio.json")
    out = tmp_path / "working" / "resultado.mp4"
    ok = aivs_wan.refine_face(cfg, [tmp_path / "r.png"], _clip(tmp_path / "wan.mp4"), out, tmp_path, report)
    assert ok is False and not out.exists()
    assert report.data["rosto"]["ok"] is False and "rosto humano" in report.data["avisos"][0]
    assert "no source face" in (tmp_path / "working" / "depuracao" / "facefusion.log").read_text()


def test_refine_face_skips_old_ffmpeg(tmp_path, monkeypatch, fake_facefusion):
    monkeypatch.setattr(aivs_wan, "ffmpeg_version", lambda: (4, 4))
    cfg = aivs_wan.Config(facefusion_dir=str(fake_facefusion), output=str(tmp_path / "working"))
    report = aivs_wan.Report(tmp_path / "working" / "relatorio.json")
    ok = aivs_wan.refine_face(cfg, [tmp_path / "r.png"], tmp_path / "wan.mp4", tmp_path / "out.mp4", tmp_path, report)
    assert ok is False and "ffmpeg 4.4" in report.data["avisos"][0]
    assert not (fake_facefusion / "args.json").exists()  # nem chega a rodar


def test_ffmpeg_version_reads_this_machine():
    version = aivs_wan.ffmpeg_version()
    assert version is None or (isinstance(version, tuple) and version >= (4, 0))
