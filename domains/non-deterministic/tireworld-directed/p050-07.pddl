(define (problem tireworld-050-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l42) (road l1 l27) (road l1 l3) (road l10 l43) (road l11 l22) (road l12 l41) (road l13 l18) (road l13 l28) (road l14 l45) (road l15 l29) (road l16 l0) (road l17 l28) (road l17 l9) (road l18 l49) (road l19 l46) (road l2 l49) (road l20 l24) (road l21 l8) (road l22 l34) (road l23 l25) (road l24 l15) (road l24 l20) (road l25 l3) (road l26 l48) (road l27 l11) (road l28 l13) (road l28 l17) (road l29 l2) (road l29 l30) (road l3 l1) (road l30 l44) (road l31 l47) (road l32 l38) (road l33 l12) (road l33 l40) (road l34 l17) (road l35 l39) (road l35 l4) (road l36 l32) (road l36 l43) (road l37 l31) (road l38 l14) (road l39 l35) (road l4 l23) (road l4 l35) (road l40 l33) (road l41 l12) (road l41 l5) (road l42 l20) (road l43 l10) (road l43 l36) (road l44 l30) (road l45 l39) (road l46 l19) (road l46 l44) (road l47 l21) (road l47 l31) (road l48 l26) (road l48 l6) (road l49 l10) (road l5 l16) (road l5 l26) (road l6 l48) (road l6 l7) (road l7 l28) (road l8 l19) (road l9 l17) (road l9 l37) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l14) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l37) (spare-in l39) (spare-in l4) (spare-in l41) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l40))
    (:goal (vehicle-at l44))
)