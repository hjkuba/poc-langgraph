---
name: document-step
description: Documenta um novo passo/feature implementado nesta POC (LangChain + LangGraph) nos três lugares onde o projeto mantém contexto — README.md, o bloco de comentário inicial de app.py e CLAUDE.md. Use depois de implementar algo novo no grafo (nó, ferramenta, forma de roteamento, etc.) ou quando o usuário pedir para "documentar" o que foi feito.
---

# Documentar um passo implementado

Este projeto é uma POC didática (`poc-langgraph`) onde cada novo conceito
(tool calling, RAG, persistência, etc.) precisa ficar rastreável em três
lugares — não só um. Depois de implementar (ou quando pedirem para
documentar) uma nova peça do grafo, atualize os três:

## 1. README.md

- Se o passo introduz um conceito novo (não só um ajuste), adicione uma
  seção própria no estilo das existentes ("Tool calling e roteamento
  condicional: como funciona", "RAG: como funciona", "Persistência: como
  funciona") — uma lista numerada explicando o mecanismo, em português,
  citando os arquivos/funções envolvidos.
- Atualize a seção "Estrutura" (árvore de arquivos) se houver arquivo novo.
- Atualize a seção "Rodando" se o comportamento observável do app mudou.
- Se o passo estava listado em "Próximos passos", remova-o de lá.

## 2. Bloco de comentário inicial de app.py

- Adicione uma linha em "O que este exemplo mostra" resumindo o passo em
  1-2 linhas (mesmo nível de detalhe dos itens existentes — o que o passo
  faz e por quê, não como).
- Atualize a lista "Estrutura (padrão src layout)" se houver arquivo novo,
  mantendo o alinhamento das setas `->`.
- Não repita neste bloco o detalhe passo a passo que já está no README —
  aqui é só o resumo de alto nível.

## 3. CLAUDE.md

- Na seção "Arquitetura", se houver arquivo novo em `src/agent/`, adicione
  um bullet descrevendo sua responsabilidade (mesmo formato dos existentes:
  **`caminho`** — o que faz, citando decisões não óbvias).
- Se o passo mudou o fluxo do grafo (`START -> ... -> END`), atualize o
  diagrama ASCII no topo da seção "Arquitetura".
- Se a implementação envolveu alguma escolha não óbvia que valha a pena um
  futuro Claude saber antes de mexer de novo (ex.: um modelo que não
  funcionou bem, um threshold calibrado, um comportamento que piorou ao
  "reforçar"), adicione um bullet em "Coisas não óbvias ao mexer nisso".
  Não duplique aqui o que já está simplesmente descrito na arquitetura.

## Ordem e consistência

Faça as três edições na mesma tarefa, não em momentos separados — os três
arquivos devem ficar consistentes entre si ao final. Se o passo for pequeno
demais para justificar uma seção nova no README (ex.: um ajuste de
parâmetro), ainda assim considere se vale ao menos uma linha no bloco de
app.py e/ou um bullet em CLAUDE.md.
