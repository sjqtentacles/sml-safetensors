fun runAllSuites () =
  ( Harness.reset ()
  ; Tests.run () )

fun main () =
  OS.Process.exit
    (if runAllSuites () then OS.Process.success else OS.Process.failure)
