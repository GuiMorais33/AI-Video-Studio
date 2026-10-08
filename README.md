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

**Situação atual: Fases 1 e 2.**

- Preparo do vídeo.
- Seleção da pessoa com cliques.
- Rastreamento com SAM 2.1.
- Correção em qualquer quadro.
- Exportação da prévia (MP4 com áudio), da máscara em vídeo e das máscaras em PNG.

O roteiro completo e o que foi revisado no plano original estão em
[docs/ROADMAP.md](docs/ROADMAP.md) e [docs/AVALIACAO-DO-PLANO.md](docs/AVALIACAO-DO-PLANO.md).

> Use apenas vídeos e rostos seus ou de quem autorizou. Veja [docs/LICENCAS.md](docs/LICENCAS.md)
> antes de usar em publicidade.

## Instalação no Windows (via WSL2)

A Meta recomenda rodar o SAM 2 no WSL com Ubuntu. O navegador continua sendo o do Windows.

1. **Diagnóstico do Windows.** No PowerShell, dentro da pasta do projeto:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\diagnose.ps1
   ```
   O script só lê informações. Ele aponta o que falta (WSL, virtualização, memória, GPU).
2. **WSL2 + Ubuntu.** Se ainda não tiver, abra o PowerShell **como administrador**, rode o comando
   abaixo e reinicie o PC:
   ```powershell
   wsl --install -d Ubuntu
   ```
3. **Dentro do Ubuntu:**
   ```bash
   sudo apt update && sudo apt install -y python3 python3-venv python3-pip git ffmpeg
   # Node.js 20+ para o painel (ex.: via nvm ou https://nodejs.org)
   git clone https://github.com/GuiMorais33/AI-Video-Studio.git ~/AI-Video-Studio
   cd ~/AI-Video-Studio
   bash scripts/setup.sh
   ```
   Clone em `~/`, não em `/mnt/c/...`: o disco do Windows visto pelo WSL é bem mais lento.

   O `setup.sh` faz o seguinte:
   - cria `engine/.venv`;
   - instala o PyTorch (só-CPU se não houver GPU NVIDIA);
   - instala o SAM 2 oficial da Meta numa versão fixa;
   - baixa o checkpoint **SAM 2.1 Tiny** (~156 MB);
   - instala o painel;
   - roda o diagnóstico.
4. **Abrir o estúdio:**
   ```bash
   bash scripts/start.sh
   ```
   Acesse <http://localhost:3000> no navegador do Windows.

Linux nativo: mesmos passos do item 3.

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
scripts/           setup.sh, start.sh, diagnose.ps1
docs/              Roteiro, avaliação do plano, licenças
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
```

`tests/test_sam2_integration.py` roda o SAM 2.1 oficial de ponta a ponta. Ele usa pesos aleatórios
e verifica o encanamento, não a qualidade. Se o SAM 2 não estiver instalado, o teste é pulado.
