(define (problem tireworld-3-1)
  (:domain tire-adl)
  (:objects n0 n1 n2 - location)
  (:init (vehicle-at n2)
         (road n0 n1) (road n1 n0)
         (road n0 n2) (road n2 n0)
         (spare-in n1)
  )
  (:goal (vehicle-at n0))
)