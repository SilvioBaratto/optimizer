function Test() -> string {
  client Model
  prompt #"
    Say hello world in Italian.

    {{ ctx.output_format }}
  "#
}