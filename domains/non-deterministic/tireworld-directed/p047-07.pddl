(define (problem tireworld-047-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l38) (road l10 l30) (road l10 l44) (road l11 l43) (road l12 l22) (road l13 l37) (road l14 l28) (road l15 l42) (road l15 l5) (road l16 l6) (road l17 l30) (road l18 l39) (road l19 l1) (road l19 l45) (road l2 l13) (road l20 l15) (road l21 l31) (road l21 l5) (road l22 l30) (road l23 l29) (road l23 l41) (road l24 l26) (road l25 l9) (road l26 l34) (road l27 l16) (road l27 l43) (road l28 l32) (road l29 l35) (road l3 l31) (road l30 l10) (road l31 l21) (road l32 l24) (road l33 l3) (road l34 l23) (road l35 l25) (road l35 l29) (road l36 l26) (road l36 l41) (road l37 l12) (road l37 l4) (road l38 l33) (road l39 l17) (road l39 l18) (road l39 l40) (road l4 l40) (road l40 l20) (road l41 l36) (road l42 l18) (road l43 l27) (road l44 l8) (road l45 l46) (road l46 l14) (road l5 l15) (road l6 l0) (road l7 l19) (road l8 l7) (road l9 l11) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l18) (spare-in l19) (spare-in l22) (spare-in l30) (spare-in l31) (spare-in l37) (spare-in l38) (spare-in l43) (spare-in l44) (spare-in l7) (spare-in l8) (vehicle-at l2))
    (:goal (vehicle-at l1))
)