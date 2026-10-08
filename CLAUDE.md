# AI Video Studio — notas para o Claude Code

- O usuário fala português (PT-BR); responda, comente e escreva mensagens de interface em PT-BR.
- Orçamento R$ 0: nunca adicione API paga, serviço com cobrança automática ou dependência que exija
  cartão.
- Uma fase só avança quando o critério em `docs/ROADMAP.md` for verificado com vídeo real. Atualize
  a coluna Status ao concluir algo.
- Hardware alvo: Windows + WSL2 (Ubuntu), provavelmente sem GPU NVIDIA. Tudo pesado precisa ter
  caminho em CPU ou rodar remotamente (Kaggle).
- Licenças importam (anúncios da VN Store): confira `docs/LICENCAS.md` antes de adicionar modelos.

## Comandos

```bash
bash scripts/setup.sh                              # instala tudo (WSL/Linux)
bash scripts/start.sh                              # motor :8765 + painel :3000
cd engine && .venv/bin/python -m pytest -q         # testes do motor
cd web && npm run typecheck && npm run build       # checagens do painel
STUDIO_TRACKER=demo engine/.venv/bin/studio serve  # motor sem IA, para mexer na interface
```

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
