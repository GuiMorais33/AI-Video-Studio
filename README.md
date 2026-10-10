# AI Video Studio

Estúdio local para transformar pessoas em vídeos reais com IA. Ferramentas gratuitas, rodando no
seu computador.

```
Vídeo original ──► SAM 2.1 (seleciona e rastreia a pessoa) ──► máscaras por quadro
                                                                     │
Referência (ChatGPT) ──► Wan 2.2 Animate, modo replacement (Fase 3) ◄┘
                                     │
                  FFmpeg: composição, áudio original, exportação MP4
```

**Situação atual: Fases 1, 2 e 3 validadas; a Fase 4 (botão do estúdio → Kaggle → resultado) funcionou de
ponta a ponta pelo motor e falta só o teste com um vídeo seu.**

- Fases 1 e 2, validadas com pesos reais do SAM 2.1 em vídeos reais:
  - preparo do vídeo;
  - seleção da pessoa com cliques;
  - rastreamento com SAM 2.1;
  - correção em qualquer quadro;
  - exportação da prévia, da máscara e dos PNGs.
- Fase 3:
  - pacote para o Wan 2.2 Animate;
  - notebook do Kaggle (GPU grátis);
  - botão **Gerar no Kaggle**, que envia, roda e traz o resultado sozinho;
  - importação com o áudio original e comparativo antes/depois.
  - Testado numa T4 grátis do Kaggle (09/10/2026): um clipe de 4,8 s em 34 min, com pose, fundo e
    iluminação fiéis ao vídeo. Tempos, qualidade e dicas em [docs/KAGGLE.md](docs/KAGGLE.md).

O roteiro completo e o que foi revisado no plano original estão em
[docs/ROADMAP.md](docs/ROADMAP.md) e [docs/AVALIACAO-DO-PLANO.md](docs/AVALIACAO-DO-PLANO.md).

> Use apenas vídeos e rostos seus ou de quem autorizou. Veja [docs/LICENCAS.md](docs/LICENCAS.md)
> antes de usar em publicidade.

## Instalação no Windows (um clique)

O estúdio roda no Ubuntu dentro do WSL2, como a Meta recomenda para o SAM 2. Você continua usando
o navegador do Windows.

**Espaço livre no disco C:** pelo menos 5 GB. Para usar uma placa NVIDIA no SAM 2 são 12 GB, porque
o PyTorch com GPU tem ~7 GB. Com menos que isso, o instalador usa a versão só-CPU (~200 MB), que
funciona igual, só que mais devagar.

1. Abra o **PowerShell** (normal, não precisa ser administrador) e cole:
   ```powershell
   irm https://raw.githubusercontent.com/GuiMorais33/AI-Video-Studio/HEAD/scripts/instalar-windows.ps1 -OutFile $env:TEMP\instalar-aivs.ps1
   powershell -ExecutionPolicy Bypass -File $env:TEMP\instalar-aivs.ps1
   ```
2. Responda ao que o instalador pedir:
   - **permissão de administrador e reinício:** só se o WSL ainda não estiver instalado. Depois de
     reiniciar, rode os mesmos comandos de novo;
   - **usuário e senha do Ubuntu:** só na primeira vez. Guarde a senha;
   - **senha do Ubuntu (`[sudo] password`):** para instalar Python, FFmpeg etc. Ela não aparece
     enquanto você digita.
3. No fim aparece o atalho **AI Video Studio** na Área de Trabalho. Ele sobe o estúdio e abre
   <http://localhost:3000> sozinho.

O instalador:

- instala o WSL2 com Ubuntu, se faltar;
- baixa o projeto em `~/AI-Video-Studio` dentro do Ubuntu;
- roda `scripts/setup.sh`, que:
  - cria `engine/.venv`;
  - instala o PyTorch:
    - com GPU quando há placa NVIDIA RTX 20 / GTX 16 ou mais nova, driver 580+ e 12 GB livres;
    - só-CPU nos outros casos, ou se o download da versão com GPU falhar. Depois de atualizar o
      driver ou liberar espaço, rodar o instalador de novo troca para a versão com GPU;
  - instala o SAM 2 oficial da Meta numa versão fixa;
  - baixa o checkpoint **SAM 2.1 Tiny** (~156 MB);
  - instala o Node.js 22 se o do sistema for antigo;
  - instala o painel;
  - roda o diagnóstico;
- cria o atalho na Área de Trabalho.

Pode rodar o instalador de novo a qualquer momento: ele continua de onde parou e atualiza o projeto.

**Diagnóstico sem instalar nada.** Com o projeto baixado, rode no PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\diagnose.ps1
```

**Linux nativo ou instalação manual no Ubuntu:**
```bash
git clone https://github.com/GuiMorais33/AI-Video-Studio.git ~/AI-Video-Studio
cd ~/AI-Video-Studio && bash scripts/setup.sh   # instala via apt o que faltar
bash scripts/start.sh                           # http://localhost:3000
```
Clone em `~/`, não em `/mnt/c/...`: o disco do Windows visto pelo WSL é bem mais lento.

## Uso

### Painel

1. **Novo projeto:** escolha o vídeo, o início e a duração (padrão 5 s, máximo 15 s).
2. **Selecionar:** clique na pessoa. O primeiro clique carrega o modelo e pode levar alguns minutos
   em CPU.
   - Botão direito ou Shift+clique exclui uma área.
   - Desfazer e Limpar tudo corrigem a seleção.
3. **Rastrear:** propaga a máscara pelo vídeo inteiro.
4. **Corrigir:** navegue com ← →. Num quadro errado, clique para incluir ou excluir e rastreie de
   novo.
5. **Exportar:** gera `preview.mp4`, `mask.mp4` e `masks.zip`, com comparativo antes/depois na
   página.
6. **Personagem (Kaggle):**
   - envie a foto de corpo inteiro (sua ou do personagem criado no ChatGPT) e, se quiser,
     descreva-o em inglês (roupa, cores, materiais);
   - envie de 3 a 6 **fotos do rosto** (de frente, meio perfil, sorrindo): no fim, o rosto do vídeo
     é trocado pelo delas (FaceFusion), para ficar igual à pessoa;
   - clique em **Gerar pacote para o Kaggle** e depois em **Gerar no Kaggle**;
   - o resultado volta com o áudio original e um comparativo lado a lado.

   Passo a passo e modo manual em [docs/KAGGLE.md](docs/KAGGLE.md).

### Linha de comando

```bash
source engine/.venv/bin/activate
studio diagnose                       # sistema, hardware e dependências
studio models list                    # checkpoints SAM 2.1 disponíveis
studio models download --model small # baixar outro tamanho
studio track video.mp4 --point 0.5,0.45 [--negative 0.7,0.3] [--frame 0] [--start 2] [--seconds 5]
studio serve                          # API local em http://127.0.0.1:8765 (docs em /docs)
```

O comando `track` faz tudo sem interface e imprime o tempo por quadro. Os pontos são normalizados
(0–1) no quadro indicado.

### Configuração (variáveis de ambiente)

| Variável | Padrão | Efeito |
|---|---|---|
| `STUDIO_DATA_DIR` | `./data` | Projetos, banco SQLite e modelos |
| `STUDIO_SAM2_MODEL` | `tiny` | `tiny`, `small`, `base_plus`, `large` |
| `STUDIO_DEVICE` | `auto` | `cpu`, `cuda`, `mps` |
| `STUDIO_MAX_SECONDS` | `5` | Duração padrão do trecho |
| `STUDIO_MAX_SIDE` | `854` | Maior lado do clipe (≈480p) |
| `STUDIO_MAX_FPS` | `30` | FPS máximo do clipe |
| `STUDIO_TRACKER` | `sam2` | `demo` simula o SAM 2 com círculos, só para testar a interface |
| `NEXT_PUBLIC_ENGINE_URL` | `http://127.0.0.1:8765` | Endereço do motor visto pelo navegador |

## Estrutura

```
engine/            Motor Python (FastAPI + SQLite + SAM 2.1 + FFmpeg)
  studio/
    media.py       FFmpeg: inspeção, recorte/redução, quadros, codificação
    tracking.py    SAM 2.1 (oficial) e rastreador demo
    session.py     Cliques, correção, desfazer e rastreamento sobre o SAM 2
    projects.py    Arquivos do projeto e exportação
    api.py         API HTTP usada pelo painel
    diagnose.py    Diagnóstico de hardware/software
  tests/           pytest (inclui teste com o SAM 2 real, se instalado)
web/               Painel Next.js
remote/kaggle/     Notebook do Wan 2.2 Animate (gerado de aivs_wan.py) e seus testes
scripts/           instalar-windows.ps1, setup.sh, start.sh, node-env.sh, diagnose.ps1
docs/              Roteiro, avaliação do plano, guia do Kaggle, licenças
data/              (gerado) projetos, studio.db, modelos — fora do Git
```

Cada projeto fica em `data/projects/<id>/`:

| Caminho | Conteúdo |
|---|---|
| `source.*` | vídeo original intacto |
| `clip.mp4` | trecho reduzido com áudio |
| `frames/` | quadros JPEG |
| `masks/obj1/` | máscaras PNG |
| `exports/` | saídas |

## Testes

```bash
cd engine && .venv/bin/python -m pytest -q
cd web && npm run typecheck && npm run build
bash scripts/tests/torch-choice.test.sh   # escolha do PyTorch (GPU simulada)
bash scripts/tests/setup.test.sh          # setup.sh de ponta a ponta com pip/GPU/disco simulados
pwsh -NoProfile -File scripts/tests/instalar-windows.test.ps1   # instalador do Windows com WSL simulado
# Notebook do Kaggle na CPU com modelos minúsculos (requer remote/kaggle/requirements-test.txt):
python -m pytest -q remote/kaggle/tests
python remote/kaggle/build_notebook.py   # regrava o .ipynb depois de mudar aivs_wan.py
```

`tests/test_sam2_integration.py` roda o SAM 2.1 oficial de ponta a ponta. Ele usa pesos aleatórios
e verifica o encanamento, não a qualidade. Se o SAM 2 não estiver instalado, o teste é pulado.

O aviso `cannot import name '_C' from 'sam2'` é esperado. Ele significa que a extensão CUDA
opcional não foi compilada. Segundo a Meta, isso só desliga um pós-processamento que tapa pequenos
buracos nas máscaras.

## Desempenho em CPU

O rastreamento com SAM 2.1 Tiny levou cerca de 2,2 s por quadro numa CPU de 4 threads. Isso dá
cerca de 5,6 min para 5 s a 30 fps, com pico de 3,2 GB de RAM. Detalhes e dicas em
[docs/ROADMAP.md](docs/ROADMAP.md).
