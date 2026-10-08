# Gerar o personagem no Kaggle (GPU grátis)

O Wan 2.2 Animate precisa de uma GPU que o seu PC não tem. O Kaggle empresta de graça uma
**GPU T4 x2**: cerca de 30 h por semana, em sessões de até 12 h. O estúdio prepara tudo e o
notebook roda lá **em lote**, sem interface web e sem túnel, que é o uso permitido.

## Uma vez só: preparar a conta

1. Crie **uma** conta em <https://www.kaggle.com>. Ter várias contas dá banimento.
2. Verifique o telefone em **Settings → Phone verification**. Sem isso, não há GPU nem internet
   no notebook.
3. Para o modo automático, gere o token em **Settings → API → Generate New Token** e copie.
   No estúdio, etapa 4, cole em **Token do Kaggle → Salvar token**.
   - O token fica só no seu computador, em `~/.kaggle/access_token` dentro do Ubuntu.

## Modo automático (recomendado)

1. No estúdio:
   - selecione e rastreie a pessoa (etapas 1 e 2);
   - envie a imagem do personagem;
   - clique em **Gerar pacote para o Kaggle**.
2. Clique em **Gerar no Kaggle**. O estúdio:
   - envia o pacote como **dataset privado** `aivs-<projeto>`;
   - roda o notebook privado `ai-video-studio-wan-animate` na T4;
   - mostra o andamento (download dos modelos, pose, geração);
   - baixa `resultado.mp4` e importa sozinho.
3. O resultado aparece na etapa 4:
   - `wan_final.mp4`: o personagem com o áudio original;
   - `comparativo.mp4`: antes e depois lado a lado.

Pode deixar a página fechada enquanto roda. Só um notebook por vez.

## Modo manual

1. No estúdio, gere o pacote e baixe `wan_package.zip`.
2. No Kaggle, em **Datasets → New Dataset**, envie o `wan_package.zip`. Deixe **privado**.
3. Em **Code → New Notebook → File → Import Notebook**, importe
   [`remote/kaggle/wan_animate_kaggle.ipynb`](../remote/kaggle/wan_animate_kaggle.ipynb).
4. No painel lateral do notebook:
   - **Accelerator:** GPU T4 x2;
   - **Internet:** on;
   - **Add Input:** o dataset do passo 2.
5. Clique em **Save Version → Save & Run All**.
6. Quando terminar, abra a aba **Output**, baixe `resultado.mp4` e importe no estúdio
   (etapa 4 → **Importar resultado**).

## O que o notebook faz

| Etapa | Detalhe |
|---|---|
| Pacote | Lê vídeo (múltiplos de 16, ex.: 832×464), máscaras do SAM 2.1, referência e `job.json` |
| Máscara | Dilatação 7×7 ×3 e blocos de 16 px, como o pré-processamento oficial; fundo com a área da pessoa em preto |
| Pose e rosto | ViTPose oficial do Wan, guiado pela caixa da **sua** máscara (acerta a pessoa certa com várias no vídeo); rosto 512×512 |
| Texto | umT5 em fp16; depois é liberado da memória |
| Geração | `WanAnimatePipeline` do diffusers 0.41.0 com: transformer GGUF Q4_K_M, LoRA lightx2v (6 passos), LoRA de reiluminação, offload em blocos, fp16 |
| Saída | `resultado.mp4`, `relatorio.json` (tempos, VRAM, avisos, erro) e prévias em `depuracao/` |

**Downloads por sessão: cerca de 28 GB.** Ficam em `/kaggle/tmp`, não na saída.

| Modelo | Tamanho |
|---|---|
| Wan2.2-Animate-14B Q4_K_M (GGUF da QuantStack) | ~11,5 GB |
| umT5 + CLIP + VAE (repositório diffusers oficial) | ~13 GB |
| ViTPose-H wholebody (repositório oficial) | ~2,5 GB |
| LoRAs (lightx2v + relight, Kijai) | ~1 GB |

Estimativa, ainda não medida numa T4: de 15 a 45 min por clipe de 5 s a 16 fps, a maior parte em
download e geração.

## Se der erro

O estúdio mostra o erro do `relatorio.json`. Os mais comuns:

| Mensagem | O que fazer |
|---|---|
| Sem GPU / `cuda` indisponível | Acelerador "GPU T4 x2" desligado, ou telefone não verificado |
| Falha de download | Internet desligada no notebook, ou Hugging Face instável: rode de novo |
| `CUDA out of memory` | Gere o pacote com **16 fps** e um trecho menor (3 s) |
| `NaN` na geração | No notebook, use `cfg.dtype = "float32"` (bem mais lento) |
| LoRA opcional ignorada | Aviso, não erro: gera sem reiluminação |

## Regras para não perder a conta

- Notebook e dataset **privados**.
- Só conteúdo apropriado: o Kaggle tem detector automático de conteúdo adulto e bane.
- Nada de túneis (ngrok, cloudflared), SSH ou interfaces web.
- Use apenas vídeos e rostos seus ou com autorização.

## O que já foi testado e o que falta

Testado aqui (CPU, sem GPU), com modelos minúsculos de pesos aleatórios e no mesmo formato dos
reais:

- o fluxo inteiro do notebook, a partir de um pacote gerado pelo estúdio;
- GGUF no formato original do Wan, com o codificador de movimento quantizado;
- LoRAs no formato da Kijai;
- o ViTPose oficial rodando um ONNX em pasta (o mesmo formato do repositório oficial);
- o notebook num kernel Jupyter;
- a automação com a API do Kaggle simulada.

**Ainda não testado numa T4 de verdade.** Isso depende da sua conta Kaggle:

- memória e tempo reais;
- estabilidade do fp16;
- qualidade do resultado.

A primeira execução é o teste da Fase 3. O `relatorio.json` traz os números para ajustarmos.
