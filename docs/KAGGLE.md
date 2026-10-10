# Gerar o personagem no Kaggle (GPU grátis)

O Wan 2.2 Animate precisa de uma GPU que o seu PC não tem. O Kaggle empresta de graça uma
**GPU T4 x2**: 30 h por semana (renova no sábado, 00:00 UTC), em sessões de até 12 h. O estúdio
prepara tudo e o notebook roda lá **em lote**, sem interface web e sem túnel, que é o uso
permitido. Na etapa 4, o estúdio mostra quantas horas você já usou na semana.

## Uma vez só: preparar a conta

1. Crie **uma** conta em <https://www.kaggle.com>. Ter várias contas dá banimento.
2. Verifique o telefone em **Settings → Phone verification**. Sem isso, não há GPU nem internet
   no notebook.
3. Para o modo automático, gere o token em **Settings → API → Generate New Token** e copie.
   No estúdio, etapa 4, cole e clique em **Salvar token**.
   - Se o Kaggle baixar um arquivo `kaggle.json` em vez de mostrar o token, abra o arquivo no
     Bloco de Notas e cole o conteúdo inteiro, com as chaves `{ }`.
   - O token fica só no seu computador, dentro do Ubuntu: em `~/.kaggle/access_token`, ou em
     `~/.kaggle/kaggle.json` no caso do arquivo.

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
| Composição | O personagem gerado é colado sobre o vídeo **original**, só dentro da máscara e com borda suave: o fundo fica intacto (o Wan redesenha o quadro inteiro e o fundo sairia levemente diferente) |
| Refino do rosto | FaceFusion 3.9.1 na 2ª T4: troca o rosto principal de cada quadro pelo das **fotos do rosto** (hyperswap, reforço de detalhe 512, todas as fotos combinadas) e restaura a nitidez (GFPGAN), respeitando mãos e cabelo na frente do rosto. O FaceFusion só vê o personagem (fora da máscara fica preto), então não troca o rosto de outra pessoa do fundo; o resultado volta a ser colado só dentro da máscara |
| Nitidez e tamanho | Real-ESRGAN 2× (pelo FaceFusion) e ajuste final ao tamanho dos Reels: **1080×1920** em pé, 1920×1080 deitado |
| Saída | `resultado.mp4` (pronto), `relatorio.json` (tempos, VRAM, avisos, erro) e prévias em `depuracao/`: `wan_bruto.mp4` (o Wan puro), `wan_sem_refino.mp4` (composto, antes do rosto) e os logs do FaceFusion |

Cada etapa do acabamento parte do vídeo anterior: se uma falhar, o resultado segue sem ela, com aviso.
Logo depois da composição já existe um `resultado.mp4` provisório, então uma queda no acabamento
não perde a geração.

O refino é o padrão. Ele fica de fora, com aviso no relatório, quando:
- as fotos não têm um rosto humano detectável (personagem robô, de capacete);
- o filtro de conteúdo adulto do FaceFusion barra o vídeo. Esse filtro fica ligado e não pega vídeo
  comum: no vídeo de dança dos testes, nenhum quadro foi marcado;
- uma etapa do FaceFusion passa de 30 min (ex.: download de modelo travado). O processo é encerrado
  e o vídeo segue sem ela.

Depois de trocar a imagem do personagem ou as fotos do rosto, **gere o pacote de novo**: o botão
"Gerar no Kaggle" fica bloqueado enquanto o pacote estiver desatualizado.

**Downloads por sessão: cerca de 30 GB.** Ficam em `/kaggle/tmp`, não na saída.

| Modelo | Tamanho |
|---|---|
| Wan2.2-Animate-14B Q4_K_M (GGUF da QuantStack) | ~11,5 GB |
| umT5 + CLIP + VAE (repositório diffusers oficial) | ~13 GB |
| ViTPose-H wholebody (repositório oficial) | ~2,5 GB |
| LoRAs (lightx2v + relight, Kijai) | ~1 GB |
| FaceFusion (hyperswap, GFPGAN, Real-ESRGAN, detectores, filtro de conteúdo) | ~1,9 GB |

**Tempo medido numa T4 (09/10/2026):** 34 min para um clipe de 4,8 s a 16 fps (77 quadros,
464×832). Com as 30 h semanais, dá para uns 50 clipes por semana.

| Etapa | Tempo |
|---|---|
| Instalação dos pacotes e ambiente | ~1,5 min |
| Pose e rosto (ViTPose na 2ª T4) | 40 s |
| Texto (download do umT5 e codificação) | 1,6 min |
| Download e carga do Wan GGUF, VAE, CLIP e LoRAs | 5–6 min |
| Geração: 6 passos de ~2,5 min cada | 15 min |
| VAE (codificar o fundo e decodificar o vídeo) e o resto da geração | ~6 min |

Memória: pico de 12,7 GB na GPU 0 com o umT5 e de 8,3 GB na geração (a T4 tem 14,6 GB).

## Qualidade e dicas

Nos testes (vídeo de uma mulher dançando na chuva, referência de um robô de suéter e gorro):

- a pose, o fundo, a chuva e a iluminação ficam muito fiéis ao vídeo original;
- a roupa e as cores vêm da imagem de referência;
- **só o Wan não segura a identidade de uma pessoa real**: o rosto sai "parecido" e segue as
  expressões do vídeo. Por isso o refino do rosto (FaceFusion) é o padrão: num teste, o rosto
  gerado virou o da foto (óculos, nariz, pele) mantendo a expressão;
- descrever o personagem em inglês (roupa, materiais, cores) deixa o resultado mais fiel: no
  teste, os braços e as mãos ficaram robóticos como na referência.

**Fotos para ficar igual a você:**
- a foto principal, de **corpo inteiro**, de frente e com fundo liso: dela vêm corpo, roupa e
  cabelo;
- **3 a 6 fotos do rosto**, nítidas e com boa luz: de frente, meio perfil, sorrindo. Sem óculos
  escuros, boné ou cabelo na frente. O refino combina todas numa identidade só;
- vídeos com alguém de porte e cabelo parecidos com os seus dão o resultado mais natural. O refino
  troca o rosto, não o formato da cabeça nem o cabelo.

## Se der erro

O estúdio mostra o erro do `relatorio.json`. Os mais comuns:

| Mensagem | O que fazer |
|---|---|
| Sem GPU / `cuda` indisponível | Acelerador "GPU T4 x2" desligado, ou telefone não verificado |
| Falha de download | Internet desligada no notebook, ou Hugging Face instável: rode de novo |
| `CUDA out of memory` | Gere o pacote com **16 fps** e um trecho menor (3 s) |
| `NaN` na geração | No notebook, use `cfg.dtype = "float32"` (bem mais lento) |
| LoRA opcional ignorada | Aviso, não erro: gera sem reiluminação |
| Refino do rosto não aplicado | Aviso, não erro: o motivo está no relatório e o log em `depuracao/facefusion.log` |

## Regras para não perder a conta

- Notebook e dataset **privados**.
- Só conteúdo apropriado: o Kaggle tem detector automático de conteúdo adulto e bane.
- Nada de túneis (ngrok, cloudflared), SSH ou interfaces web.
- Use apenas vídeos e rostos seus ou com autorização.

## O que já foi testado e o que falta

Testado numa T4 de verdade (09/10/2026), disparado pelo motor do estúdio com as mesmas funções do
botão **Gerar no Kaggle**: envio do pacote, execução, progresso ao vivo, download e importação do
resultado. Os problemas encontrados nas primeiras execuções já foram corrigidos no notebook (versões
de pacotes da imagem do Kaggle, memória da GPU e uso das duas T4).

Testado também aqui, na CPU, com modelos minúsculos no mesmo formato dos reais: o fluxo inteiro do
notebook, o GGUF, as LoRAs, o ViTPose e a automação com a API do Kaggle simulada.

Falta:

- rodar com um vídeo seu e um personagem do ChatGPT, pelo botão do estúdio no seu PC;
- avaliar se 4 passos (mais rápido) mantêm a qualidade.
