(define (problem tireworld-050-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l3) (road l1 l5) (road l10 l33) (road l11 l20) (road l11 l21) (road l12 l40) (road l13 l10) (road l13 l41) (road l14 l25) (road l15 l2) (road l15 l22) (road l15 l36) (road l15 l6) (road l16 l47) (road l17 l23) (road l18 l41) (road l19 l28) (road l19 l45) (road l2 l15) (road l2 l35) (road l20 l11) (road l20 l43) (road l21 l11) (road l21 l27) (road l22 l3) (road l23 l17) (road l23 l4) (road l24 l7) (road l25 l8) (road l26 l15) (road l26 l38) (road l27 l38) (road l28 l0) (road l28 l19) (road l29 l12) (road l29 l46) (road l3 l22) (road l3 l30) (road l3 l34) (road l30 l3) (road l30 l31) (road l31 l30) (road l32 l11) (road l33 l42) (road l34 l21) (road l34 l39) (road l35 l2) (road l35 l44) (road l36 l0) (road l37 l49) (road l38 l26) (road l39 l32) (road l39 l34) (road l4 l46) (road l40 l24) (road l41 l13) (road l42 l37) (road l43 l20) (road l43 l31) (road l44 l35) (road l44 l47) (road l45 l19) (road l45 l48) (road l46 l29) (road l47 l44) (road l48 l45) (road l49 l48) (road l5 l17) (road l6 l1) (road l6 l15) (road l7 l24) (road l7 l9) (road l8 l18) (road l8 l25) (road l9 l14) (road l9 l7) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l14) (spare-in l15) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l24) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l4) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l6) (vehicle-at l16))
    (:goal (vehicle-at l38))
)