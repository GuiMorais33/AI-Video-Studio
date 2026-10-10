# Licenças

Uso declarado (10/2026): o estúdio **não é usado para anúncios pagos**. Por isso entram também
modelos com licença não comercial ou de pesquisa, que dão o melhor resultado (principalmente o
refino do rosto). A coluna "Uso comercial" fica registrada para o caso de isso mudar: aí os itens
marcados como **Não** precisariam ser trocados ou licenciados. Isto não é aconselhamento jurídico.

| Componente | Licença | Uso comercial | Observação |
|---|---|---|---|
| SAM 2.1 (código e checkpoints) | Apache-2.0 | Sim | Repositório oficial `facebookresearch/sam2` |
| FFmpeg | LGPL/GPL (conforme a build) | Sim, para uso interno | Builds com libx264 são GPL; não redistribua binários sem cumprir a GPL |
| Next.js, React, FastAPI, NumPy, Pillow | MIT/BSD | Sim | |
| PyTorch | BSD-3 | Sim | |
| Wan 2.2 Animate (pesos e código, inclusive o ViTPose do pré-processamento) | Apache-2.0 | Sim | |
| GGUF do Wan2.2-Animate (QuantStack) | Derivado do Apache-2.0 | Sim | Conversão dos pesos oficiais |
| LoRA lightx2v (reempacotada pela Kijai) | Apache-2.0 (conferir na origem lightx2v) | Provável | |
| LoRA de reiluminação (Kijai/Wan) | A confirmar | A confirmar | Opcional: o notebook gera sem ela |
| diffusers, transformers, peft, accelerate | Apache-2.0 | Sim | |
| YOLOv10 (detector do pré-processamento oficial) | AGPL-3.0 | Evitado | Não precisamos: a caixa da pessoa vem da máscara do SAM 2.1, que acerta mais |
| ComfyUI | GPL-3.0 | Sim, como ferramenta | Não usado |

## Refino do rosto (FaceFusion 3.9.1, padrão no notebook)

| Componente | Licença | Uso comercial | Papel |
|---|---|---|---|
| FaceFusion (código) | OpenRAIL-AS | Com restrições de uso | Roda em lote no Kaggle, versão fixa (commit `72470819`) |
| hyperswap_1a_256 | ResearchRAIL | **Não** | Troca o rosto (256 px, com reforço de detalhe 512) |
| arcface_w600k_r50 (InsightFace) | Não comercial | **Não** | Lê a identidade das fotos; todos os trocadores dependem dele |
| gfpgan_1.4 | Apache-2.0 | Sim | Restaura a nitidez do rosto |
| yolo_face, xseg_1 | GPL-3.0 | Sim, como ferramenta | Detecta o rosto e o que está na frente dele (mãos, cabelo) |
| 2dfan4, fan_68_5, bisenet, fairface | Conferir | A confirmar | Pontos do rosto e análise |
| Filtro de conteúdo (nsfw_1/2/3) | Apache-2.0 / MIT | Sim | Embutido no FaceFusion e **mantido ligado**: barra só conteúdo adulto |
| inswapper_128, alphaface_256 (alternativas) | Não comercial | **Não** | Trocadores alternativos (`cfg.face_swapper_model`) |
| codeformer, gpen_bfr (alternativas) | Não comercial | **Não** | Restauradores alternativos (`cfg.face_enhancer_model`) |

## Pessoas e plataformas

- Use vídeos e rostos próprios ou de quem autorizou. Pôr o rosto de alguém num vídeo sem
  autorização pode violar direito de imagem e a LGPD.
- Conteúdo realista alterado por IA deve ser marcado com o rótulo de IA do Instagram/Meta.
- Imagens geradas no ChatGPT seguem os termos da OpenAI (não gerar marcas ou celebridades).
