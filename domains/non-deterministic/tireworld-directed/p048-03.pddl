(define (problem tireworld-048-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l14) (road l1 l44) (road l1 l9) (road l10 l20) (road l10 l29) (road l11 l6) (road l12 l20) (road l12 l8) (road l13 l15) (road l14 l35) (road l15 l46) (road l16 l31) (road l17 l33) (road l18 l7) (road l19 l42) (road l2 l17) (road l20 l10) (road l21 l30) (road l21 l45) (road l22 l47) (road l22 l8) (road l23 l21) (road l24 l27) (road l25 l47) (road l26 l38) (road l27 l13) (road l28 l29) (road l28 l39) (road l29 l28) (road l30 l21) (road l30 l6) (road l31 l40) (road l32 l22) (road l32 l43) (road l33 l16) (road l34 l37) (road l35 l18) (road l36 l24) (road l37 l25) (road l37 l41) (road l37 l45) (road l38 l2) (road l39 l36) (road l4 l0) (road l40 l5) (road l41 l26) (road l42 l34) (road l43 l32) (road l44 l1) (road l44 l43) (road l45 l21) (road l45 l37) (road l46 l19) (road l47 l22) (road l5 l9) (road l6 l30) (road l6 l4) (road l7 l23) (road l7 l43) (road l8 l12) (road l8 l22) (road l8 l3) (road l9 l1) (spare-in l0) (spare-in l14) (spare-in l15) (spare-in l18) (spare-in l20) (spare-in l22) (spare-in l25) (spare-in l28) (spare-in l32) (spare-in l35) (spare-in l38) (spare-in l4) (spare-in l43) (spare-in l46) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l11))
    (:goal (vehicle-at l3))
)