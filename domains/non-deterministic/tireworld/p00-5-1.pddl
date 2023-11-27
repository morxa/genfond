(define (problem tireworld-5-1)
  (:domain tire-adl)
  (:objects n0 n1 n2 n3 n4 n5 - location)
  (:init (vehicle-at n0)
         (road n0 n1) (road n1 n0)
         (road n1 n2) (road n2 n1)
         (road n2 n5) (road n5 n2)
         (road n0 n3) (road n3 n0)
         (spare-in n3)
         (road n3 n4) (road n4 n3)
         (road n4 n5) (road n5 n4)
         (spare-in n4)
  )
  (:goal (vehicle-at n5))
)
