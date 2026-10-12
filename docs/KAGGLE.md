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
3. Logo em seguida, **aqui no seu computador**, o estúdio troca o rosto pelo das suas fotos
   (veja [Refino do rosto](#refino-do-rosto-no-seu-computador)).
4. O resultado aparece na etapa 4:
   - `wan_final.mp4`: o vídeo final em 1080×1920 (em pé), com o áudio original;
   - `comparativo.mp4`: antes e depois lado a lado.

Pode deixar a página fechada enquanto roda, mas o estúdio precisa continuar aberto no computador.
Só um notebook por vez: se ele já estiver rodando (por exemplo, disparado de outro computador
com o mesmo token), o estúdio avisa e não envia outro.

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
   (etapa 4 → **Importar resultado**). O refino do rosto começa sozinho depois da importação.

## O que o notebook faz

| Etapa | Detalhe |
|---|---|
| Pacote | Lê vídeo (múltiplos de 16, ex.: 832×464), máscaras do SAM 2.1, referência e `job.json` |
| Máscara | Dilatação 7×7 ×3 e blocos de 16 px, como o pré-processamento oficial; fundo com a área da pessoa em preto |
| Pose e rosto | ViTPose oficial do Wan, guiado pela caixa da **sua** máscara (acerta a pessoa certa com várias no vídeo); rosto 512×512 |
| Texto | umT5 em fp16; depois é liberado da memória |
| Geração | `WanAnimatePipeline` do diffusers 0.41.0 com: transformer GGUF Q4_K_M, LoRA lightx2v (6 passos), LoRA de reiluminação, offload em blocos, fp16 |
| Composição | O personagem gerado é colado sobre o vídeo **original**, só dentro da máscara e com borda suave: o fundo fica intacto (o Wan redesenha o quadro inteiro e o fundo sairia levemente diferente) |
| Saída | `resultado.mp4` (composto, no tamanho da geração, ex.: 464×832), `relatorio.json` (tempos, VRAM, avisos, erro) e prévias em `depuracao/`: `wan_bruto.mp4` (o Wan puro), pose, rosto e máscara |

O notebook **não** troca o rosto: isso é feito no seu computador, logo depois (próxima seção).

Depois de trocar a imagem do personagem, **gere o pacote de novo**: o botão "Gerar no Kaggle" fica
bloqueado enquanto o pacote estiver desatualizado. As fotos do rosto não vão para o Kaggle.

## Refino do rosto no seu computador

O Kaggle **bloqueia o FaceFusion e ferramentas parecidas de troca de rosto**. Um funcionário do
Kaggle confirmou em 18/08/2026 (["FaceFusion and similar are blocked"](https://www.kaggle.com/discussions/product-feedback/735585)):
a sessão é encerrada por fora, sem erro, e aparece como `cancel_acknowledged`. A revisão automática
do Kaggle também pode banir a conta por "Resource abuse". Por isso o refino roda no estúdio, no seu
computador, e nunca no Kaggle.

Assim que o vídeo do Kaggle é importado, o estúdio começa o refino sozinho:

1. só o personagem vai para o FaceFusion 3.9.1 (fora da máscara fica preto), então ninguém do fundo
   tem o rosto trocado;
2. o rosto de cada quadro vira o das **fotos do rosto** (hyperswap com reforço de detalhe 512, todas
   as fotos combinadas numa identidade) e a nitidez é restaurada (GFPGAN), respeitando mãos e cabelo
   na frente do rosto;
3. o rosto refinado é colado de volta só dentro da máscara, com borda suave;
4. o vídeo vai para o tamanho dos Reels (**1080×1920** em pé, 1920×1080 deitado) e recebe o áudio
   original.

Enquanto o refino roda, `wan_final.mp4` já existe sem ele. Quando termina, o arquivo é trocado pela
versão refinada.

- **Tempo:** roda na CPU, então é mais lento que o Kaggle: cerca de 10 min para 5 s de vídeo (77
  quadros) num processador de 4 núcleos, mais em notebooks mais simples.
- **Primeira vez:** baixa cerca de 1,7 GB de modelos do FaceFusion (do GitHub).
- **Requisito:** Ubuntu 24.04 no WSL (Python 3.11+ e FFmpeg 5.1+). No 22.04 o instalador avisa e o
  estúdio funciona sem o refino; atualize com `sudo do-release-upgrade` e rode o instalador de novo.
- **Trocou as fotos do rosto?** Clique em **Refazer o refino do rosto**. Não precisa gerar no Kaggle
  de novo.

O refino fica de fora, com aviso na etapa 4 e o vídeo sem ele, quando:
- as fotos não têm um rosto humano detectável (personagem robô, de capacete);
- o filtro de conteúdo adulto do FaceFusion barra o vídeo. Esse filtro fica ligado e não pega vídeo
  comum: no vídeo de dança dos testes, nenhum quadro foi marcado;
- o FaceFusion passa 20 min sem dar sinal de vida (ex.: download de modelo travado). O processo é
  encerrado;
- o sistema encerra o FaceFusion por falta de memória. Feche outros programas ou aumente a memória do
  WSL (`memory=` no `.wslconfig`).

O log completo fica em `data/projects/<projeto>/exports/facefusion_rosto.log`.

**Downloads por sessão: cerca de 30 GB.** Ficam em `/kaggle/tmp`, não na saída.

| Modelo | Tamanho |
|---|---|
| Wan2.2-Animate-14B Q4_K_M (GGUF da QuantStack) | ~11,5 GB |
| umT5 + CLIP + VAE (repositório diffusers oficial) | ~13 GB |
| ViTPose-H wholebody (repositório oficial) | ~2,5 GB |
| LoRAs (lightx2v + relight, Kijai) | ~1 GB |

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
| O Kaggle cancelou a execução (`cancel_acknowledged`) | Não é erro do notebook. Alguém clicou em Cancel/Stop session no kaggle.com, ou o próprio Kaggle parou a sessão (limite de tempo ou ferramenta bloqueada). A mensagem mostra a última etapa e o motivo, quando o Kaggle informa |
| Já há uma execução deste notebook rodando | Espere terminar (talvez tenha sido disparada de outro computador) ou pare em kaggle.com |
| Refino do rosto não aplicado | Aviso, não erro: o motivo aparece na etapa 4 e o log fica em `exports/facefusion_rosto.log` |

## Regras para não perder a conta

- Notebook e dataset **privados**.
- Só conteúdo apropriado: o Kaggle tem detector automático de conteúdo adulto e bane.
- Nada de túneis (ngrok, cloudflared), SSH ou interfaces web.
- **Nada de FaceFusion ou outras ferramentas de troca de rosto no Kaggle:** são bloqueadas e podem
  levar ao banimento. O estúdio faz essa parte no seu computador.
- Use apenas vídeos e rostos seus ou com autorização.

## O que já foi testado e o que falta

Testado numa T4 de verdade (09/10/2026), disparado pelo motor do estúdio com as mesmas funções do
botão **Gerar no Kaggle**: envio do pacote, execução, progresso ao vivo, download e importação do
resultado. Os problemas encontrados nas primeiras execuções já foram corrigidos no notebook (versões
de pacotes da imagem do Kaggle, memória da GPU e uso das duas T4).

Testado também aqui, na CPU, com modelos minúsculos no mesmo formato dos reais: o fluxo inteiro do
notebook, o GGUF, as LoRAs, o ViTPose e a automação com a API do Kaggle simulada.

Em 10/2026, a primeira execução com o FaceFusion no notebook terminou cancelada pelo Kaggle, sem
erro, como acontece com ferramentas de troca de rosto bloqueadas. O FaceFusion saiu do notebook e o
refino passou para o computador.

Falta:

- rodar com um vídeo seu e as suas fotos, pelo botão do estúdio no seu PC, com o refino local;
- avaliar se 4 passos (mais rápido) mantêm a qualidade.
