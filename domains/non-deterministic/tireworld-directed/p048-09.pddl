(define (problem tireworld-048-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l33) (road l1 l20) (road l1 l26) (road l10 l45) (road l11 l42) (road l12 l37) (road l13 l24) (road l14 l11) (road l15 l13) (road l16 l0) (road l17 l37) (road l18 l38) (road l19 l36) (road l2 l35) (road l20 l1) (road l20 l43) (road l21 l19) (road l21 l31) (road l22 l40) (road l23 l20) (road l24 l5) (road l25 l47) (road l26 l4) (road l27 l29) (road l28 l15) (road l29 l44) (road l3 l7) (road l30 l6) (road l31 l21) (road l32 l28) (road l33 l39) (road l34 l27) (road l35 l22) (road l36 l3) (road l37 l23) (road l37 l34) (road l38 l41) (road l39 l17) (road l4 l2) (road l40 l16) (road l41 l14) (road l42 l31) (road l43 l46) (road l44 l25) (road l45 l32) (road l46 l9) (road l47 l8) (road l5 l26) (road l6 l17) (road l6 l46) (road l7 l10) (road l8 l18) (road l9 l12) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l27) (spare-in l3) (spare-in l33) (spare-in l34) (spare-in l36) (spare-in l37) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l43) (spare-in l44) (spare-in l5) (spare-in l6) (spare-in l9) (vehicle-at l30))
    (:goal (vehicle-at l29))
)