(define (problem triangle-tireworld-handcrafted2)
  (:domain triangle-tire)
  (:objects l0 l1_1 l1_2 l1_3 l2_1 l2_2 l2_3 l2_4 l2_5 l_g - location)
  (:init
    (not-flattire)
    (vehicle-at l0)
    (road l0 l1_1) (road l1_1 l1_2) (road l1_2 l1_3) (road l1_3 l_g)
    (road l0 l2_1) (road l2_1 l2_2) (road l2_2 l2_3) (road l2_3 l2_4) (road l2_4 l_g) (road l2_5 l_g)
    (spare-in l1_1)
    (spare-in l1_2)
    (spare-in l2_1)
    (spare-in l2_2)
    (spare-in l2_3)
    (spare-in l2_4)
  )
  (:goal (vehicle-at l_g))
)