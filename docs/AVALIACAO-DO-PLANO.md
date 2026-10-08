# Avaliação do plano original (out/2026)

O plano foi escrito no ChatGPT. Esta é uma revisão técnica feita antes de seguirmos além da Fase 2.
Os dados de cotas e modelos mudam rápido: confira as fontes antes de tomar decisões.

## Veredito

**A direção está certa, mas a ordem das fases e a fonte de GPU gratuita precisam mudar.** O critério
de qualidade também precisa ser medido mais cedo.

## O que se mantém

| Decisão do plano | Avaliação |
|---|---|
| Wan 2.2 Animate (modo replacement) como motor de troca de personagem | Continua sendo o principal modelo **aberto** para substituir um personagem em vídeo real. Licença Apache-2.0, então o uso comercial é permitido. |
| SAM 2.1 para isolar e rastrear a pessoa | Correto. O modo replacement do Wan precisa de máscara da pessoa e de fundo; as máscaras que geramos alimentam essa etapa. |
| FFmpeg para recorte, composição e áudio | Correto. Gratuito e padrão da indústria. |
| Validar cada etapa antes de avançar; R$ 0; sem API paga automática | Correto. Mantido como regra do projeto. |
| Next.js + Python + SQLite, sem Redis/Kubernetes | Adequado para uso local por uma pessoa. |

## O que muda

### 1. Testar o risco principal primeiro

A incerteza que decide o projeto é se o Wan Animate roda de graça com qualidade aceitável. No plano
original ela só aparece na Fase 4, depois de construir SAM 2 e FaceFusion. Se ela falhar, a
arquitetura muda. **Proposta:** fazer o teste do Wan logo após a Fase 2 (agora numerado como
Fase 3), antes de qualquer integração de troca facial.

### 2. Kaggle no lugar do ZeroGPU

- **ZeroGPU (Hugging Face), conta gratuita:**
  - A documentação oficial lista cerca de **3,5 min/dia**; algumas fontes falam em 5.
  - Há relatos de um limite de cerca de **3 execuções por dia**.
  - Uma RTX 4090 leva cerca de **450 s** para 5 s de vídeo em 720×1280, mais que a cota diária
    inteira.
  - Serve no máximo para uma demonstração.
- **Kaggle Notebooks:**
  - Cerca de **30 h de GPU por semana** (2× T4 de 16 GB ou 1× P100), em sessões de até 12 h.
  - É uma ordem de grandeza a mais. O número exato aparece nas configurações da conta.
- **Caminho técnico:**
  - Rodar o Wan 2.2 Animate 14B **quantizado em GGUF** no **ComfyUI sem interface**.
  - Guias da comunidade apontam cerca de 16 GB de VRAM como piso viável, com descarregamento
    para a RAM.
  - **Isso ainda precisa ser medido no próprio Animate.**
- **Alternativa:** a Modal anuncia US$ 30/mês em créditos no plano Starter.
  - Exige conta, e às vezes cartão. Só seria usada com limite de gasto zerado.

### 3. Não reimplementar a geração

O ComfyUI já tem fluxos oficiais do Wan 2.2 Animate "Mix" (replacement), carregadores GGUF, LoRA de
4 passos e de reiluminação. O estúdio deve **orquestrar**, não reescrever a inferência:

- **Local:** seleção da pessoa (SAM 2), referências do ChatGPT, composição, revisão e exportação.
- **Remoto (Kaggle):** um fluxo ComfyUI fixo, executado sem interface gráfica.
- **Segurança:** usar só nós amplamente usados (ComfyUI-GGUF, KJNodes, controlnet_aux), com versões
  fixadas.

### 4. FaceFusion fora do caminho principal

- O Wan Animate já transfere as expressões do rosto original no modo replacement.
- **Licenças:**
  - O código do FaceFusion é OpenRAIL-AS.
  - O inswapper exige licença comercial da InsightFace.
  - Não encontramos a licença dos pesos do hyperswap.
- Para anúncio da VN Store isso é risco jurídico.
- Fica como etapa **opcional** de refino, só com modelo de licença comercial confirmada.

### 5. Etapas que faltavam

- **Upscale:** o Wan gera em 480p/720p; Reels são 1080×1920.
- **Rótulo de IA no Instagram:** conteúdo realista alterado por IA deve ser marcado.
- **Consentimento:** de quem aparece no vídeo. Evitar o rosto de pessoas reais sem autorização.
- **Wan-Animate-2 (ago/2026):** pesos públicos. Ainda não confirmamos se tem modo replacement.
  Fica como candidato a comparar.

### 6. Medir a qualidade antes de investir

Passar um clipe por um demo hospedado do Wan 2.2 Animate (gratuito) e comparar com o Reel de
referência. Se o modelo em si não chegar perto, nenhuma engenharia em volta resolve.

## Roteiro revisado

Ver [ROADMAP.md](ROADMAP.md).

## Fontes

- ZeroGPU, documentação oficial: https://huggingface.co/docs/hub/spaces-zerogpu
- Fórum HF sobre a cota gratuita: https://discuss.huggingface.co/t/what-is-the-free-zerogpu-quota-for-1-space/178610
- Cotas do Kaggle e outras GPUs gratuitas (set/2026): https://gpuperhour.com/blog/free-cloud-gpus-and-credits
- Wan 2.2 (repositório oficial): https://github.com/Wan-Video/Wan2.2
- Wan2.2-Animate-2-14B: https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B
- ComfyUI, tutorial do Wan 2.2 Animate: https://docs.comfy.org/tutorials/video/wan/wan2-2-animate
- GGUF do Animate-14B: https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF
- Tempo de geração numa RTX 4090: https://www.nextdiffusion.ai/tutorials/how-to-use-wan-2-2-animate-in-comfyui-for-character-animations
- VRAM do modo Mix: https://artokun.mintlify.app/blog/wan-animate-comfyui
- Licença do FaceFusion: https://cdn.jsdelivr.net/gh/facefusion/facefusion@master/README.md
- Modal (créditos gratuitos): https://freetier.co/directory/products/modal
