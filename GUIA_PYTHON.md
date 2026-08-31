# Guia de Python para quem vem de TypeScript

Este guia explica a sintaxe e os conceitos de Python usados nesta POC,
sempre com o paralelo em TypeScript ao lado. Não é um curso de Python
genérico — cada exemplo vem de um arquivo real deste projeto, então dá
pra ler este guia com o código aberto ao lado.

## Sumário

1. [Diferenças de filosofia](#1-diferenças-de-filosofia)
2. [Ambiente e dependências](#2-ambiente-e-dependências)
3. [Módulos e imports](#3-módulos-e-imports)
4. [Tipos e `TypedDict`](#4-tipos-e-typeddict)
5. [Funções, type hints e docstrings](#5-funções-type-hints-e-docstrings)
6. [Coleções: listas, tuplas, dicts e sets](#6-coleções-listas-tuplas-dicts-e-sets)
7. [Strings e f-strings](#7-strings-e-f-strings)
8. [Controle de fluxo](#8-controle-de-fluxo)
9. [Comprehensions](#9-comprehensions)
10. [Funções como valores, closures e decorators](#10-funções-como-valores-closures-e-decorators)
11. [Classes](#11-classes)
12. [Async/await](#12-asyncawait)
13. [Tratamento de erros](#13-tratamento-de-erros)
14. [Idiomas específicos do Python](#14-idiomas-específicos-do-python)
15. [Tabela de referência rápida](#15-tabela-de-referência-rápida)

---

## 1. Diferenças de filosofia

Antes da sintaxe, três diferenças que explicam *por que* o código Python
deste projeto parece diferente de TypeScript, mesmo quando faz a mesma
coisa:

- **Tipagem é opcional e não é imposta em runtime.** Em TypeScript, o
  compilador (`tsc`) recusa código que não bate com os tipos, e o
  resultado (JS puro) não carrega tipo nenhum. Em Python, as anotações de
  tipo (`def somar(a: float, b: float) -> float:`) são **só documentação
  e uma ajuda pra ferramentas** (VSCode, mypy) — o interpretador não
  verifica nada disso ao rodar. Você pode chamar `somar("a", "b")` e o
  Python só vai reclamar quando `a + b` falhar de verdade, não antes.
- **Blocos são por indentação, não por chaves.** Não existe `{` `}` nem
  `;`. O nível de indentação (4 espaços, por convenção) *é* a sintaxe.
  Esquecer de indentar direito é erro de sintaxe, não só de estilo.
- **Convenção de nomes é `snake_case`, não `camelCase`.** Funções,
  variáveis e módulos: `retrieval_node`, `messages_before`. Classes:
  `PascalCase` (`StructuredTool`), igual TS. Constantes em
  `SCREAMING_SNAKE_CASE`: `SIMILARITY_THRESHOLD`, `MODEL_NAME`.

## 2. Ambiente e dependências

| Node/TS | Python | Neste projeto |
|---|---|---|
| `package.json` | `requirements.txt` (ou `pyproject.toml`) | [`requirements.txt`](requirements.txt) |
| `node_modules/` (automático) | `.venv/` (você cria explicitamente) | `.venv/`, git-ignorado |
| `npm install` | `pip install -r requirements.txt` | mesmo comando |
| `npx`/scripts do `package.json` | `python app.py` direto | `python app.py` |

A diferença mais importante: Node isola dependências por projeto
automaticamente (`node_modules/` local). Python **não** faz isso por
padrão — `pip install` sem um venv ativo instala pacotes globalmente pra
todo o sistema. Por isso todo projeto Python decente começa com:

```bash
python -m venv .venv        # cria o ambiente isolado (pasta .venv/)
source .venv/bin/activate   # "entra" no ambiente nesta sessão do terminal
pip install -r requirements.txt
```

`source .venv/bin/activate` é conceitualmente parecido com rodar
`npx <alguma-cli-local>` — troca o `python`/`pip` do terminal pelos
binários de dentro de `.venv/`, em vez dos globais.

## 3. Módulos e imports

```python
# src/agent/graph.py
from typing import Optional
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from .configuration import checkpointer, llm      # import relativo
from .mcp_tools import mcp_tools
```

```typescript
import type { Optional } from "some-lib"; // Python não precisa de "type"
import { AIMessage, SystemMessage, ToolMessage } from "langchain-core/messages";
import { checkpointer, llm } from "./configuration"; // import relativo
import { mcpTools } from "./mcpTools";
```

Pontos de atenção:

- `from .configuration import x` — o `.` na frente é import **relativo**
  ao pacote atual (`src/agent/`), igual `./configuration` em TS/JS. Sem o
  `.`, seria um pacote instalado (`from langchain_core.messages import
  ...` busca a lib `langchain-core` instalada via pip).
- Não existe `export`. Tudo que está no nível do módulo (função, classe,
  variável) já é importável de fora — o mais perto de TS seria como se
  **tudo** fosse `export` automaticamente. Convenção pra "privado":
  prefixo `_` (`_LOCAL_TOOL_NAMES`, `_context_message`) — é só uma
  convenção lida por humanos e por `from modulo import *`, não uma
  restrição real: nada impede `import agent.graph; agent.graph._context_message(...)`.
- `src/agent/` vira o pacote `agent` porque tem `pyproject.toml`/`pip
  install -e .` (modo editável) — mecanismo diferente de `tsconfig.json`
  paths ou workspaces do npm, mas resolve o mesmo problema (importar
  `agent.graph` de qualquer lugar sem paths relativos gigantes).

## 4. Tipos e `TypedDict`

```python
# src/agent/state.py
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: list[str]
    approval_decision: str
    last_tool_call_signature: Optional[str]
    tool_call_repeat_count: int
```

```typescript
interface State {
  messages: Message[];              // sem equivalente direto ao Annotated
  context: string[];
  approvalDecision: string;
  lastToolCallSignature: string | null;
  toolCallRepeatCount: number;
}
```

- `TypedDict` é o que mais se parece com uma `interface` do TS: descreve o
  formato de um **dicionário** (não uma classe de verdade — não tem
  métodos, não instancia com `new`). `state["messages"]` continua sendo
  acesso de dict normal (colchetes + string), não `state.messages`.
- `Optional[str]` == `string | null` (ou `| undefined`) em TS. Python só
  tem `None` (não existe `undefined` separado de `null`), então
  `Optional[X]` é sempre só `X | None`.
- `list[str]` (minúsculo) é a forma moderna (Python 3.9+) de dizer "lista
  de strings" — equivalente a `string[]`. Código mais antigo usa
  `List[str]` (maiúsculo, importado de `typing`); são a mesma coisa.
- `Annotated[list, add_messages]` não tem paralelo direto em TS — é um
  jeito de "grudar" metadado extra num tipo (aqui, uma função reducer que
  o LangGraph lê em runtime pra saber como combinar atualizações de
  estado). Pense nele como um tipo + uma anotação que uma lib específica
  vai inspecionar depois — mais parecido com um decorator de campo do
  que com um tipo puro.
- **De novo: nada disso é validado em runtime.** Se um nó devolver
  `{"tool_call_repeat_count": "três"}` (string em vez de int), Python não
  reclama na hora — só quando algo tentar fazer `3 + "três"` depois.

## 5. Funções, type hints e docstrings

```python
# src/agent/tools.py
@tool
def consultar_saldo(conta: str) -> float:
    """Consulta o saldo de uma conta bancária ('corrente' ou 'poupança')."""
    conta_normalizada = conta.strip().lower()
    if conta_normalizada not in _SALDOS:
        raise ValueError(
            f"Conta '{conta}' não encontrada. Contas disponíveis: "
            f"{', '.join(_SALDOS)}."
        )
    return _SALDOS[conta_normalizada]
```

```typescript
function consultarSaldo(conta: string): number {
  /** Consulta o saldo de uma conta bancária ('corrente' ou 'poupança'). */
  const contaNormalizada = conta.trim().toLowerCase();
  if (!(contaNormalizada in _SALDOS)) {
    throw new Error(
      `Conta '${conta}' não encontrada. Contas disponíveis: ${Object.keys(_SALDOS).join(", ")}.`
    );
  }
  return _SALDOS[contaNormalizada];
}
```

- `def nome(param: Tipo) -> TipoRetorno:` é a assinatura tipada — visualmente
  parecida com TS, só que o tipo vem **depois** do nome do parâmetro
  (`a: float`, igual TS) mas o tipo de retorno vem depois de `->`, não `:`.
- **A docstring** (string entre `"""..."""` logo após o `def`) não é um
  comentário — é um valor de verdade, acessível em runtime via
  `consultar_saldo.__doc__`. Neste projeto ela cumpre um papel especial:
  o decorator `@tool` (próxima seção) lê essa docstring e a usa como a
  descrição que o LLM recebe pra decidir quando chamar a função. É o
  equivalente funcional de um JSDoc `/** ... */`, mas *lido pelo programa*,
  não só por ferramentas de editor.
- `raise ValueError(...)` == `throw new Error(...)`. Python tem várias
  classes de exceção nativas (`ValueError`, `TypeError`, `KeyError`...)
  em vez do `Error` genérico do JS — mais parecido com ter várias
  subclasses de `Error` já prontas.
- Parâmetros com valor default: `def retrieve(query: str, k: int = 3)`
  (`retrieval.py`) == `function retrieve(query: string, k: number = 3)`.

## 6. Coleções: listas, tuplas, dicts e sets

Python tem **quatro** tipos de coleção embutidos onde TS geralmente usa
só `Array` e `object`/`Map`. Essa é uma das maiores diferenças de
vocabulário entre as duas linguagens.

### Listas (`list`) — como `Array`

```python
tools = [somar, multiplicar, consultar_saldo]   # tools.py
chunks: list[str] = []
chunks.extend(splitter.split_text(...))          # retrieval.py
```
```typescript
const tools = [somar, multiplicar, consultarSaldo];
const chunks: string[] = [];
chunks.push(...splitter.splitText(...));
```

### Tuplas (`tuple`) — sem equivalente direto; mais perto de um array fixo/readonly

```python
graph.invoke({"messages": [("user", user_input)]}, config=config)  # app.py
```
Aqui `("user", user_input)` é uma **tupla** de 2 posições — um par
fixo, imutável, que o LangChain reconhece como `(role, content)`. Não dá
pra fazer `.append()` numa tupla. O mais próximo em TS é uma tupla
tipada: `const pair: [string, string] = ["user", userInput];` — TS tem
tuplas de verdade (só que como um caso especial de `Array`), Python tem
um **tipo separado**.

### Dicionários (`dict`) — como `object` ou `Map`

```python
_SALDOS = {"corrente": 1250.75, "poupança": 4830.10}   # tools.py
call["name"], call["args"]                              # graph.py
```
```typescript
const SALDOS: Record<string, number> = { corrente: 1250.75, poupança: 4830.10 };
call.name, call.args   // ou call["name"] se for um Record
```
Acesso é sempre com colchetes + chave: `dicionario["chave"]`, nunca
`dicionario.chave` (isso é atributo de objeto, outra coisa — ver seção
de classes). `.get("chave")` devolve `None` se não existir, em vez de
lançar erro — parecido com `map.get(k)` do `Map` de JS (que devolve
`undefined`), mas em cima de um `dict`, não de um `Map`.

### Sets (`set`) — como `Set`, com sintaxe de chaves

```python
_LOCAL_TOOL_NAMES = {t.name for t in tools}      # graph.py — set comprehension
if user_input.lower() in {"sair", "exit", "quit"}:  # app.py
```
```typescript
const LOCAL_TOOL_NAMES = new Set(tools.map(t => t.name));
if (["sair", "exit", "quit"].includes(userInput.toLowerCase())) { ... }
```
`{...}` com valores soltos (sem `:`) é um **set**, não um dict — coleção
sem ordem garantida e sem duplicatas, ótima pra checar
`"x" in conjunto` em tempo constante (mais rápido que `.includes()` numa
lista/array grande).

### Indexação e slicing — só Python

```python
state["messages"][-1]              # última mensagem
state["messages"][messages_before:]  # tudo a partir do índice messages_before
```
```typescript
messages.at(-1);          // TS/JS moderno (ES2022)
messages.slice(messagesBefore);
```
Índice negativo (`[-1]` = último item) funciona direto nos colchetes em
Python; em JS/TS precisa do método `.at()`. `lista[a:b]` (slice) é
sintaxe nativa de colchetes em Python — em TS é sempre um método,
`.slice(a, b)`.

## 7. Strings e f-strings

```python
f"Conta '{conta}' não encontrada. Contas disponíveis: {', '.join(_SALDOS)}."
f"[aprovação necessária para chamar '{pending['tool']}']"
```
```typescript
`Conta '${conta}' não encontrada. Contas disponíveis: ${Object.keys(SALDOS).join(", ")}.`
`[aprovação necessária para chamar '${pending.tool}']`
```
`f"..."` (f-string) == template literal com crase. Mesma ideia
(interpolação com `{expressão}` em vez de `${expressão}`), incluindo
poder chamar métodos dentro (`{', '.join(_SALDOS)}`). Reparem na inversão
de sujeito: em Python é `separador.join(lista)`; em TS é
`lista.join(separador)`.

Strings normais aceitam aspas simples ou duplas, sem diferença
(`'texto'` == `"texto"`), diferente de TS onde aspas simples/duplas são
strings normais e crase é reservada pra template literals.

## 8. Controle de fluxo

```python
if context := state.get("context"):        # graph.py — walrus operator
    messages.append(_context_message(context))
```
Não existe equivalente direto em TS pra isso. `:=` (operador "morsa",
Python 3.8+) atribui **e** avalia numa única expressão — aqui, guarda o
resultado de `state.get("context")` em `context` e, no mesmo fôlego, usa
esse valor na condição do `if`. Sem o walrus, teria que ser duas linhas:
```python
context = state.get("context")
if context:
    ...
```
que é exatamente como ficaria em TS: `const context = state.get("context"); if (context) { ... }`.

```python
for message in state["messages"][messages_before:]:
    for call in getattr(message, "tool_calls", None) or []:
        print(f"[tooling: {call['name']}({call['args']})]")
```
```typescript
for (const message of messages.slice(messagesBefore)) {
  for (const call of message.toolCalls ?? []) {
    console.log(`[tooling: ${call.name}(${JSON.stringify(call.args)})]`);
  }
}
```
`for x in iteravel:` é sempre "for...of" — não existe o `for (let i=0;...)`
clássico como forma idiomática (dá pra fazer com `range(len(x))`, mas é
raro). `getattr(obj, "nome", default)` é acesso de atributo dinâmico com
fallback — o parente mais próximo em TS é `obj?.nome ?? default`, mas
`getattr` funciona com o *nome do atributo como string*, então é mais
parecido com `obj["nome"] ?? default` de fato dinâmico.

**Truthiness**: `and`/`or`/`not` são palavras, não `&&`/`||`/`!`.
`None`, `0`, `""`, `[]`, `{}` e `set()` são todos "falsy" (igual JS trata
`null`/`0`/`""` como falsy) — `[] or []` devolve o segundo `[]`, mesma
lógica de curto-circuito do `||` em JS.

`while True:` (app.py) — loop infinito, igual `while (true) {}`.
`break`/`continue` funcionam igual TS.

## 9. Comprehensions

Esta é a maior mudança de idioma entre as duas linguagens. Onde TS usa
`.map()`/`.filter()` encadeados, Python usa uma sintaxe própria dentro
de colchetes/chaves:

```python
[doc.page_content for doc, score in results if score >= SIMILARITY_THRESHOLD]
```
```typescript
results
  .filter(([doc, score]) => score >= SIMILARITY_THRESHOLD)
  .map(([doc, score]) => doc.pageContent)
```

```python
{t.name for t in tools}                       # set comprehension
[_as_sync_tool(t) for t in asyncio.run(_client.get_tools())]   # list comprehension
```
```typescript
new Set(tools.map(t => t.name));
(await client.getTools()).map(asSyncTool);
```

A leitura é sempre: `[EXPRESSÃO for ITEM in ITERÁVEL if CONDIÇÃO]` — "pra
cada ITEM em ITERÁVEL, se CONDIÇÃO, produza EXPRESSÃO". `{...}` no lugar
de `[...]` faz a mesma coisa mas produz um `set`; `{k: v for ...}` produz
um `dict`. `for doc, score in results` já desempacota cada tupla
`(doc, score)` da lista `results` direto nas variáveis — equivalente ao
destructuring `([doc, score]) => ...` do exemplo TS acima.

## 10. Funções como valores, closures e decorators

Funções são cidadãs de primeira classe em Python, igual em JS/TS —
passadas como argumento, retornadas, atribuídas a variáveis, sem
sintaxe especial. O grafo inteiro é montado assim:

```python
graph_builder.add_conditional_edges(
    "chatbot", route_tools, {"approval_local": "approval_local", ...}
)
```
`route_tools` aqui é passada **sem** os parênteses `()` — é a função em
si, não o resultado de chamá-la. Igual passar `someHandler` em vez de
`someHandler()` como callback em TS.

### Closures

```python
# mcp_tools.py
def _as_sync_tool(async_tool: BaseTool) -> StructuredTool:
    def call(**kwargs):
        return asyncio.run(async_tool.ainvoke(kwargs))
    return StructuredTool(..., func=call)
```
```typescript
function asSyncTool(asyncTool: BaseTool): StructuredTool {
  function call(kwargs: Record<string, unknown>) {
    return runSync(asyncTool.ainvoke(kwargs));
  }
  return new StructuredTool({ ..., func: call });
}
```
Mesmo conceito de closure que em JS: `call` "lembra" de `async_tool` do
escopo em que foi criada, mesmo sendo devolvida e chamada depois em
outro lugar.

`**kwargs` no parâmetro de `call` significa "aceite qualquer quantidade
de argumentos nomeados, empacotados num dict chamado `kwargs`" — sem
equivalente 1:1 em TS (o mais perto é um único parâmetro objeto
`(kwargs: Record<string, unknown>)`, já que TS/JS não tem uma convenção
nativa de "argumentos nomeados arbitrários" separada de "um objeto").
Existe também `*args` (posicionais arbitrários), não usado neste projeto
mas comum em Python — equivalente ao rest parameter `...args` do JS.

### Decorators

```python
@tool
def somar(a: float, b: float) -> float:
    """Soma dois números."""
    return a + b
```
```typescript
@Tool()
somar(a: number, b: number): number {
  return a + b;
}
```
Um decorator "embrulha" a função/classe logo abaixo dele, tipicamente
devolvendo uma versão modificada ou registrando ela em algum lugar — aqui,
`@tool` (do LangChain) transforma a função Python comum
`somar` num objeto `StructuredTool` que sabe seu próprio nome, descrição
(a docstring) e schema de argumentos (os type hints), pronto pra ser
oferecido a um LLM. Já `@mcp.tool()` (`mcp_server.py`) leva parênteses —
é uma "fábrica de decorators": `mcp.tool()` primeiro devolve o decorator
de verdade, que só então é aplicado à função. Ambos os padrões existem em
TS/JS também (decorators experimentais, usados em NestJS/Angular:
`@Injectable()` sempre leva parênteses, por exemplo) — a ideia é a mesma,
"metaprogramação" que roda quando o módulo é carregado, antes de qualquer
chamada normal de função.

## 11. Classes

Este projeto usa poucas classes próprias (a maioria das classes vem de
libs — `ChatOllama`, `StructuredTool`, `FastMCP`), mas vale saber ler:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```
```typescript
interface State {
  messages: Message[];
}
```
`TypedDict` é uma classe especial que, ao ser "herdada" (`class State(
TypedDict)`), não cria instâncias de verdade — é só uma forma de
descrever o formato de um dict comum, como uma `interface`. Uma classe
Python "de verdade" (não usada neste projeto, mas útil saber) seria:

```python
class Pessoa:
    def __init__(self, nome: str, idade: int):  # equivalente ao constructor
        self.nome = nome
        self.idade = idade

    def cumprimentar(self) -> str:
        return f"Oi, eu sou {self.nome}"

p = Pessoa("Ana", 30)
p.cumprimentar()
```
```typescript
class Pessoa {
  nome: string;
  idade: number;
  constructor(nome: string, idade: number) {
    this.nome = nome;
    this.idade = idade;
  }
  cumprimentar(): string {
    return `Oi, eu sou ${this.nome}`;
  }
}
const p = new Pessoa("Ana", 30);
p.cumprimentar();
```
Diferenças principais: `__init__` é o construtor (nome fixo, sempre esse,
diferente do `constructor` de TS que também é um nome fixo mas
"embutido" na sintaxe da classe). E **todo método recebe `self` como
primeiro parâmetro explícito** — o equivalente ao `this` implícito do
TS, só que em Python você escreve `self` na assinatura de cada método
(`def cumprimentar(self)`) e o Python passa a instância automaticamente
quando você chama `p.cumprimentar()`. Esquecer o `self` é um erro comum
de quem vem de outra linguagem.

## 12. Async/await

```python
# mcp_tools.py
async def _connect(): ...
mcp_tools = [_as_sync_tool(t) for t in asyncio.run(_client.get_tools())]
```
```typescript
async function connect() { ... }
const mcpTools = (await client.getTools()).map(asSyncTool);
```
`async def` e `await` existem em Python com a sintaxe quase idêntica a
JS/TS — mesma ideia de função que pode pausar em pontos de I/O sem
bloquear. A diferença grande é **onde vive o event loop**: em Node, todo
o programa já roda dentro de um event loop desde o início (você só usa
`await` onde precisar). Em Python, código async só roda dentro de um
event loop explicitamente iniciado — `asyncio.run(minha_coroutine())` é o
jeito mais comum de "entrar" nesse loop a partir de código síncrono
comum. Sem estar dentro de um `asyncio.run()` (ou equivalente), `await`
não funciona.

Isso teve consequência real neste projeto: como o resto do código
(`app.py`, `configuration.py`) é síncrono, carregar as tools MCP
(inerentemente assíncronas) exigiu usar `asyncio.run()` pontualmente pra
"entrar e sair" do mundo async só na hora de buscar as tools — ver o
comentário em [`src/agent/mcp_tools.py`](src/agent/mcp_tools.py) e o
bullet sobre isso em [`CLAUDE.md`](CLAUDE.md).

## 13. Tratamento de erros

```python
try:
    conta_normalizada = conta.strip().lower()
    if conta_normalizada not in _SALDOS:
        raise ValueError(f"Conta '{conta}' não encontrada.")
except ValueError as e:
    print(f"Erro: {e}")
```
```typescript
try {
  const contaNormalizada = conta.trim().toLowerCase();
  if (!(contaNormalizada in SALDOS)) {
    throw new Error(`Conta '${conta}' não encontrada.`);
  }
} catch (e) {
  console.log(`Erro: ${e}`);
}
```
`try`/`except` == `try`/`catch` (nome diferente, mesmo mecanismo).
`except ValueError as e:` captura só esse tipo específico de erro —
parecido com fazer `catch (e) { if (!(e instanceof ValueError)) throw e; ... }`
em TS, só que nativo da sintaxe. Não tem `finally`? Tem — `finally:`,
igual TS.

## 14. Idiomas específicos do Python

Alguns padrões que aparecem neste projeto e não têm equivalente direto:

```python
if __name__ == "__main__":
    main()
```
Todo módulo Python tem uma variável mágica `__name__`. Quando o arquivo é
executado diretamente (`python app.py`), `__name__` vale `"__main__"`;
quando é *importado* por outro módulo, vale o nome do módulo
(`"app"`). Esse `if` é como dizer "só rode isso se este arquivo foi o
ponto de entrada, não se alguém só importou ele" — resolve um problema
que em Node se resolve de outro jeito (CommonJS: `require.main === module`;
ESM: normalmente nem se coloca lógica de execução solta no topo do
arquivo, só se exporta funções e um outro arquivo chama).

```python
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / ".sqlite"
```
`__file__` é outra variável mágica: o caminho do arquivo atual —
parecido com `import.meta.url` em ESM ou `__dirname` em CommonJS. E o
operador `/` entre `Path`s **não é divisão** — a classe `Path` sobrescreve
o operador `/` pra significar "junte esse pedaço de caminho", então
`Path("a") / "b" / "c.txt"` monta `a/b/c.txt` de um jeito multiplataforma
(equivalente a `path.join("a", "b", "c.txt")` do Node, só que com
sintaxe de operador em vez de função). Isso é possível porque Python
deixa classes redefinirem o que operadores como `/`, `+`, `==` fazem
(chamado *operator overloading*) — não existe em TS/JS.

```python
tools = [*local_tools, *mcp_tools]
llm.bind_tools([*tools, *mcp_tools])
```
`*lista` dentro de outro literal de lista **espalha** os itens — mesma
sintaxe e mesmo efeito do spread `...array` em JS/TS
(`[...localTools, ...mcpTools]`).

## 15. Tabela de referência rápida

| Conceito | Python | TypeScript |
|---|---|---|
| Bloco de código | indentação | `{ }` |
| Fim de instrução | quebra de linha | `;` (opcional em TS/JS também) |
| Variável/função | `snake_case` | `camelCase` |
| Constante | `NOME_MAIUSCULO` | `NOME_MAIUSCULO` ou `const` |
| "Nada"/"vazio" | `None` | `null` / `undefined` |
| Verdadeiro/falso | `True` / `False` | `true` / `false` |
| E / ou / não | `and` / `or` / `not` | `&&` / `\|\|` / `!` |
| Interface/formato de dict | `class X(TypedDict):` | `interface X { }` |
| Union de tipos | `Optional[str]`, `str \| None` | `string \| null` |
| Lista | `list[str]`, `[1, 2, 3]` | `string[]`, `[1, 2, 3]` |
| Tupla | `(1, "a")` | `[1, "a"]` (tupla tipada) |
| Objeto/mapa | `dict`, `{"a": 1}` | `object`/`Map`, `{a: 1}` |
| Conjunto | `set`, `{1, 2, 3}` | `Set`, `new Set([1,2,3])` |
| Template string | `f"{x}"` | `` `${x}` `` |
| Loop `for...of` | `for x in iteravel:` | `for (const x of iteravel)` |
| Índice do fim | `lista[-1]` | `arr.at(-1)` |
| Fatiar lista | `lista[a:b]` | `arr.slice(a, b)` |
| Comprehension | `[f(x) for x in xs if cond]` | `xs.filter(cond).map(f)` |
| Spread | `[*a, *b]` | `[...a, ...b]` |
| Args nomeados livres | `**kwargs` | objeto único como parâmetro |
| Args posicionais livres | `*args` | `...args` |
| Try/catch | `try/except/finally` | `try/catch/finally` |
| Lançar erro | `raise ValueError("msg")` | `throw new Error("msg")` |
| Construtor de classe | `def __init__(self, ...):` | `constructor(...)` |
| `this` | `self` (explícito em cada método) | `this` (implícito) |
| Async | `async def` / `await` | `async function` / `await` |
| Entrar no event loop | `asyncio.run(fn())` | (já roda por padrão no Node) |
| Atribuir dentro de condição | `if x := f():` | não existe (usa 2 linhas) |
| Ambiente isolado | `.venv/` + `pip` | `node_modules/` + `npm` |
| Ponto de entrada | `if __name__ == "__main__":` | roda direto, ou `require.main` |
