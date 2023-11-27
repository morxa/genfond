(define (problem tireworld-4-1)
  (:domain tire-adl)
  (:objects n0 n1 n2 n3 - location)
  (:init (vehicle-at n0)
         (road n0 n1) (road n1 n0)
         (road n1 n2) (road n2 n1)
         (road n2 n3) (road n3 n2)
         (spare-in n1)
         (spare-in n2)
  )
  (:goal (vehicle-at n3))
)
