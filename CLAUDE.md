# AI Video Studio — notas para o Claude Code

- O usuário fala português (PT-BR); responda, comente e escreva mensagens de interface em PT-BR.
- Orçamento R$ 0: nunca adicione API paga, serviço com cobrança automática ou dependência que exija
  cartão.
- Uma fase só avança quando o critério em `docs/ROADMAP.md` for verificado com vídeo real. Atualize
  a coluna Status ao concluir algo.
- Hardware alvo: Windows + WSL2 (Ubuntu), provavelmente sem GPU NVIDIA. Tudo pesado precisa ter
  caminho em CPU ou rodar remotamente (Kaggle).
- Uso declarado pelo usuário (10/2026): não é para anúncios pagos, e a meta é o rosto final ficar
  igual às fotos enviadas. Modelos com licença não comercial ou de pesquisa são aceitos; registre a
  licença de cada um em `docs/LICENCAS.md`. Filtros de segurança das ferramentas (ex.: o filtro de
  conteúdo adulto do FaceFusion) não são desligados.

## Comandos

```bash
bash scripts/setup.sh                              # instala tudo (WSL/Linux)
bash scripts/start.sh                              # motor :8765 + painel :3000
cd engine && .venv/bin/python -m pytest -q         # testes do motor
cd web && npm run typecheck && npm run build       # checagens do painel
STUDIO_TRACKER=demo engine/.venv/bin/studio serve  # motor sem IA, para mexer na interface
pwsh -NoProfile -File scripts/tests/instalar-windows.test.ps1  # instalador do Windows com WSL simulado
bash scripts/tests/setup.test.sh && bash scripts/tests/torch-choice.test.sh  # setup.sh com pip/GPU/disco simulados
```

- Scripts `.ps1`:
  - só ASCII (o PowerShell 5.1 lê UTF-8 sem BOM como ANSI);
  - lembre que nomes de variáveis não diferenciam maiúsculas (`$distro` == `$Distro`);
  - comandos nativos que precisam de console (sudo, progresso) são chamados direto, fora de
    funções e sem capturar a saída.

## Arquitetura

- `engine/studio/session.py`:
  - As máscaras em disco são a fonte da verdade.
  - O estado do SAM 2 vive só na memória. Ao reconstruir, os quadros com cliques viram condição
    via `add_new_mask` com a máscara do disco.
  - Desfazer restaura o PNG salvo em `history/<click_id>.png`.
- Tarefas longas (rastrear, exportar) rodam em um único worker (`jobs.py`), com progresso no
  SQLite. Cliques retornam 409 enquanto há tarefa ativa.
- O painel (`web/`) fala direto com o motor via CORS (`NEXT_PUBLIC_ENGINE_URL`).
- Fase 3:
  - `remote/kaggle/aivs_wan.py` é a fonte do notebook. Depois de editá-lo, rode
    `python remote/kaggle/build_notebook.py` (há um teste que falha se o `.ipynb` estiver
    desatualizado).
  - O motor envia o pacote ao Kaggle por `engine/studio/kaggle_remote.py`, com tarefas `kaggle`
    numa fila separada.
  - No Kaggle: só execução em lote, privada, sem túnel e sem interface.
  - Depois do Wan, o notebook refina o rosto com o FaceFusion 3.9.1 (commit fixo em `Config`),
    usando as fotos de `faces/` do pacote (ou a referência). Se falhar, entrega o vídeo do Wan e
    registra um aviso no relatório.
