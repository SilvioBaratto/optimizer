function Test() -> string {
  client Tier1
  prompt #"
    Say hello world in Italian.

    {{ ctx.output_format }}
  "#
}