(define (problem triangle-tireworld-002)
    (:domain triangle-tire)
    (:requirements :non-deterministic :strips :typing)
    (:objects l0_0 l0_2 l0_4 l0_6 l0_8 l1_1 l1_3 l1_5 l1_7 l2_2 l2_4 l2_6 l3_3 l3_5 l4_4 - location)
    (:init (not-flattire) (road l0_0 l0_2) (road l0_0 l1_1) (road l0_2 l0_4) (road l0_2 l1_3) (road l0_4 l0_6) (road l0_4 l1_5) (road l0_6 l0_8) (road l0_6 l1_7) (road l1_1 l0_2) (road l1_1 l2_2) (road l1_3 l0_4) (road l1_3 l2_4) (road l1_5 l0_6) (road l1_5 l2_6) (road l1_7 l0_8) (road l2_2 l1_3) (road l2_2 l2_4) (road l2_2 l3_3) (road l2_4 l1_5) (road l2_4 l2_6) (road l2_4 l3_5) (road l2_6 l1_7) (road l3_3 l2_4) (road l3_3 l4_4) (road l3_5 l2_6) (road l4_4 l3_5) (spare-in l1_1) (spare-in l1_3) (spare-in l1_5) (spare-in l1_7) (spare-in l2_2) (spare-in l2_6) (spare-in l3_3) (spare-in l3_5) (spare-in l4_4) (vehicle-at l0_0))
    (:goal (vehicle-at l0_8))
)