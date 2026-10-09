# Roteiro

Regra do projeto: **uma fase só termina quando o critério de conclusão for verificado com um vídeo
real**. Orçamento: R$ 0, sem API paga automática.

A ordem abaixo já incorpora a [avaliação do plano original](AVALIACAO-DO-PLANO.md). As mudanças em
relação ao plano original são duas. O teste do Wan Animate veio para a Fase 3. A troca facial
(FaceFusion) virou opcional.

| Fase | Entrega | Critério de conclusão | Status |
|---|---|---|---|
| 1. Ambiente | Estrutura do projeto, instalador, diagnóstico de hardware | `scripts/diagnose.ps1` e `studio diagnose` rodam no PC do usuário | **Concluída:** o instalador de um clique (que roda o `studio diagnose` no fim) rodou no Windows do usuário em 09/10/2026 |
| 2. SAM 2.1 | Selecionar pessoa com clique, rastrear, corrigir, exportar prévia | Rastrear uma pessoa por 5 s e corrigir as máscaras, com vídeo real | **Validada com pesos reais em 2 vídeos reais** (ver abaixo); vale repetir com um vídeo seu |
| 3. Teste do Wan Animate | Notebook Kaggle que roda o Wan 2.2 Animate (replace) com diffusers, usando nossas máscaras ([guia](KAGGLE.md)) | Um clipe de 3–5 s com personagem substituído, gerado sem custo; tempo e qualidade anotados | Notebook pronto e testado na CPU com modelos minúsculos; **falta a 1ª execução numa T4 real (sua conta Kaggle)** |
| 4. Integração | Envio do clipe + máscara + referência do ChatGPT para o Kaggle e retorno do resultado ao estúdio | Fluxo de ponta a ponta disparado pelo estúdio | Implementada (botão "Gerar no Kaggle", API simulada nos testes); valida junto com a Fase 3 |
| 5. Refinamento | Composição com bordas suaves, ajuste de cor, upscale para 1080×1920, áudio original | Vídeo final pronto para Reels com comparativo antes/depois | A fazer |
| 6. Interface definitiva | Biblioteca de personagens, histórico, fila de gerações | Fluxo aprovado executado só pela interface | A fazer |
| Opcional | Refino de rosto (FaceFusion ou similar) | Somente com modelo de licença comercial confirmada | Em espera |

## Fase 2 — como validar com vídeo real

1. Gravar ou escolher um vídeo de 5 s com uma pessoa (sua ou autorizada).
2. Teste sem interface:
   ```bash
   engine/.venv/bin/studio track meu_video.mp4 --point 0.5,0.45
   ```
   `--point` é a posição da pessoa no primeiro quadro, de 0 a 1 (0.5,0.45 = centro, um pouco acima).
3. Abrir `data/projects/<id>/exports/preview.mp4` e verificar se o contorno acompanha a pessoa.
4. No painel (`bash scripts/start.sh`), corrigir um quadro ruim com clique de exclusão, rastrear de
   novo e exportar.
5. Anotar o tempo por quadro que o comando imprime: é a medida de desempenho da CPU.

### Validação com pesos reais (out/2026)

Feita no ambiente de nuvem do Claude Code com o checkpoint SAM 2.1 Tiny. Os sites da Meta e do
Hugging Face estavam bloqueados ali. Foi usada a cópia em meia precisão publicada pela Ultralytics
no GitHub; as 471 camadas são idênticas às do modelo oficial.

**Vídeo vertical de 5 s (mulher dançando de capa amarela)**
- 3 cliques no quadro 0.
- A máscara acompanhou a pessoa nos 120 quadros: giros, cabelo voando, braços para cima, dedos.

**Vídeo de demonstração oficial do SAM 2 (duas crianças pulando na cama)**
- 2 cliques na menina.
- Ela foi rastreada nos 200 quadros e o menino ficou de fora.

**Correção**
- Um clique de exclusão num quadro já rastreado tira só a região clicada (ex.: a saia).
- Um clique de inclusão devolve uma parte que faltava (ex.: um braço) e mantém 99–100% do resto.
- Desfazer restaura a máscara exata.
- Esse teste revelou um problema: depois de reiniciar o motor, um clique de exclusão sozinho fazia o
  SAM apagar a máscara inteira. Agora o rastreador acrescenta pontos-âncora tirados do miolo da
  máscara anterior.

### Desempenho medido (CPU, sem GPU)

SAM 2.1 Tiny, Xeon de 4 threads, clipe de 5 s em 854×480 (150 quadros):

| Etapa | Tempo |
|---|---|
| Preparar o clipe | 1,4 s |
| Primeiro clique (carrega os quadros) | 4,7 s |
| Cliques seguintes | menos de 0,1 s |
| Rastreamento | 2,2 a 2,9 s por quadro, cerca de 5,6 min no total |
| Exportação | 2 s |

Pico de RAM: 3,2 GB.

O tempo do rastreamento é proporcional ao número de quadros. Para ir mais rápido em CPU:

- `STUDIO_MAX_FPS=15` corta o tempo pela metade.
- Um trecho de 3 s em vez de 5 s também reduz o total.

O custo de cálculo é o mesmo com os pesos reais: os números valem como estimativa. Num PC de
notebook comum, espere algo entre 3 e 10 minutos por clipe de 5 s.

## Fase 3 — como foi montada

A pesquisa (out/2026) comparou três rotas: diffusers, ComfyUI sem interface e o repositório oficial
do Wan. A escolhida foi o **diffusers 0.41.0**:

- é um script Python puro, sem servidor nem interface (o perfil de menor risco no Kaggle);
- dá para testar na CPU com modelos minúsculos;
- carrega o transformer em GGUF Q4_K_M, que cabe na T4 com offload em blocos.

O repositório oficial não roda na T4 sem modificações, porque exige FlashAttention-2. O ComfyUI
fica como rota reserva.

**Mudanças em relação ao pré-processamento oficial:**
- A pose usa a caixa da máscara do SAM 2.1 em vez do detector YOLOv10. Acerta a pessoa escolhida
  em vídeos com várias pessoas e evita a licença AGPL do YOLO.
- A máscara vem do estúdio (já corrigida por você). Recebe a mesma dilatação do oficial e blocos
  de 16 px, como no template oficial do ComfyUI.

**Para a 1ª execução real:**
- siga o [guia do Kaggle](KAGGLE.md);
- traga o `relatorio.json`: tempos, VRAM, avisos e erro, se houver.
