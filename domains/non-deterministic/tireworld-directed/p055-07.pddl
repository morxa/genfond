(define (problem tireworld-055-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l6 l7 l8 l9 - location)
    (:init (road l0 l36) (road l1 l24) (road l10 l13) (road l11 l21) (road l12 l4) (road l13 l42) (road l14 l35) (road l15 l14) (road l16 l27) (road l17 l19) (road l18 l9) (road l19 l13) (road l2 l21) (road l20 l16) (road l21 l0) (road l21 l10) (road l22 l8) (road l23 l31) (road l24 l25) (road l25 l12) (road l26 l38) (road l27 l39) (road l28 l48) (road l29 l1) (road l3 l43) (road l3 l44) (road l30 l18) (road l31 l5) (road l32 l28) (road l33 l17) (road l34 l42) (road l34 l54) (road l35 l50) (road l36 l11) (road l36 l41) (road l37 l53) (road l38 l20) (road l39 l22) (road l4 l12) (road l4 l49) (road l40 l47) (road l41 l52) (road l42 l13) (road l42 l34) (road l43 l45) (road l44 l3) (road l45 l7) (road l46 l51) (road l47 l37) (road l48 l33) (road l49 l15) (road l5 l26) (road l50 l46) (road l51 l44) (road l52 l35) (road l53 l23) (road l54 l2) (road l54 l6) (road l6 l41) (road l7 l30) (road l8 l29) (road l9 l32) (spare-in l1) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l31) (spare-in l32) (spare-in l35) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l41) (spare-in l44) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l8) (vehicle-at l40))
    (:goal (vehicle-at l3))
)