"""Mede os rostos de cada quadro com o próprio FaceFusion (roda com o Python dele, não com o do motor).

    <python do FaceFusion> facefusion_probe.py <pasta do FaceFusion> <vídeo> <saída.json> <threads>

Grava, para cada quadro, a lista de rostos achados: caixa, confiança do detector e dos 68 pontos e
`r5`, a posição do nariz entre os olhos (0,5 = de frente; perto de 0 ou 1 = três quartos; bem fora
de 0..1 = de perfil). Roda depois do headless-run, quando os modelos já estão baixados. Usa os
mesmos modelos e limites da troca (face_refine.facefusion_command).
"""

import json
import os
import subprocess
import sys


def main() -> None:
    facefusion_dir, video, output, threads = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    sys.path.insert(0, facefusion_dir)
    os.chdir(facefusion_dir)
    import numpy
    from facefusion import state_manager
    from facefusion.face_creator import get_many_faces

    for key, value in {
        "face_detector_model": "yolo_face", "face_detector_size": "640x640", "face_detector_margin": [0, 0, 0, 0],
        "face_detector_angles": [0], "face_detector_score": 0.5, "face_landmarker_model": "2dfan4",
        "face_landmarker_score": 0.5, "execution_device_ids": [0], "execution_providers": ["cpu"],
        "execution_thread_count": threads, "download_providers": ["github"],
    }.items():
        state_manager.init_item(key, value)

    size = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                           "-of", "csv=p=0", video], capture_output=True, text=True, check=True).stdout.strip()
    width, height = (int(v) for v in size.split(",")[:2])
    # Quadros em BGR, como o FaceFusion lê (pelo ffmpeg, sem depender do OpenCV para ler vídeo).
    reader = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                              stdout=subprocess.PIPE)
    frames = []
    while True:
        data = reader.stdout.read(width * height * 3)
        if len(data) < width * height * 3:
            break
        frame = numpy.frombuffer(data, numpy.uint8).reshape(height, width, 3)
        faces = []
        for face in get_many_faces([frame]):
            left_eye, right_eye, nose = face.landmark_set.get("5/68")[:3]
            span = float(right_eye[0] - left_eye[0])
            faces.append({
                "box": [round(float(v), 1) for v in face.bounding_box],
                "detector": round(float(face.score_set.get("detector", 0)), 3),
                "landmarker": round(float(face.score_set.get("landmarker", 0)), 3),
                "r5": round(float(nose[0] - left_eye[0]) / span, 3) if abs(span) > 1e-6 else 99.0,
            })
        frames.append(faces)
        print(f"probe: {len(frames)}", flush=True)
    reader.stdout.close()
    reader.wait()
    with open(output, "w") as f:
        json.dump(frames, f)


if __name__ == "__main__":
    main()
