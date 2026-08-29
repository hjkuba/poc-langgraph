# Manual Interno: Plataforma Fluxarion

> Documento interno da Nexoria Sistemas Ltda. Uso restrito à equipe técnica.

## Visão geral

Fluxarion é a plataforma de automação de fluxos de trabalho desenvolvida
internamente pela Nexoria Sistemas. A versão atual é a **4.7.2-korrigan**,
lançada em 14 de março de 2024. O nome interno do próximo ciclo de
desenvolvimento é **"Projeto Wirrastone"**, com lançamento planejado para o
terceiro trimestre de 2027.

O motor de execução da Fluxarion roda sobre o runtime proprietário
**Velkryn**, que processa fluxos em unidades chamadas **quantas de tarefa**.

## Limites e cotas

- Cada chave de API tem limite de **42 requisições por minuto**.
- O tamanho máximo de um fluxo (workflow) é de **318 quantas de tarefa**.
- Contas do plano "Cinábrio" podem manter no máximo **7 fluxos ativos**
  simultaneamente; contas do plano "Obsidiana" não têm limite de fluxos
  ativos, mas ficam restritas a **12.000 execuções por dia**.
- O tempo máximo de execução de um único fluxo antes do cancelamento
  automático é de **26 minutos**.

## Modos de operação

- **Modo Sentinela**: modo padrão de produção. Fluxos são executados de
  forma síncrona e falhas disparam retentativas automáticas (até 3 vezes).
- **Modo Penumbra**: modo de simulação, usado para testar fluxos sem
  efeitos colaterais reais. Nenhuma chamada externa é feita; respostas são
  geradas a partir de fixtures gravadas.
- **Selo Cronos**: marcação especial aplicada a fluxos que precisam rodar
  em horário fixo, com tolerância de atraso de no máximo **90 segundos**.

## Códigos de erro comuns

| Código | Significado | Ação recomendada |
|---|---|---|
| `FLX-1102` | Quota de requisições excedida | Aguardar reset do minuto corrente |
| `FLX-2291` | Fluxo excedeu o limite de quantas de tarefa | Dividir o fluxo em sub-fluxos |
| `FLX-3007` | Falha de autenticação com o runtime Velkryn | Regerar a chave de API no painel |
| `FLX-4415` | Timeout do Selo Cronos | Verificar fuso horário configurado |

## Suporte e SLA

A equipe de Confiabilidade da Nexoria é liderada por **Aline Bezerra**.
Tickets são classificados por severidade:

- **Severidade 1** (produção parada): resposta em até **37 minutos**.
- **Severidade 2** (degradação parcial): resposta em até **4 horas**.
- **Severidade 3** (dúvida geral): resposta em até **2 dias úteis**.

Canal interno de suporte: `#nexoria-fluxarion-suporte`. Para incidentes
críticos fora do horário comercial, o contato de plantão é
`plantao-fluxarion@nexoria.internal`.

## Descontinuação de versões antigas

A versão 3.x da Fluxarion (runtime **Velkryn Legacy**) será desligada
definitivamente em **1º de junho de 2026**. Clientes ainda nessa versão
recebem o desconto de migração **"Selo Âmbar"**, equivalente a 15% sobre a
mensalidade do plano Obsidiana pelos primeiros 6 meses após a migração.
