(define (problem tireworld-048-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l17) (road l1 l19) (road l10 l7) (road l11 l35) (road l12 l43) (road l13 l21) (road l14 l23) (road l15 l30) (road l16 l12) (road l17 l29) (road l18 l8) (road l19 l7) (road l2 l14) (road l20 l46) (road l21 l27) (road l22 l11) (road l23 l14) (road l23 l25) (road l24 l0) (road l24 l3) (road l25 l44) (road l26 l0) (road l27 l20) (road l28 l4) (road l29 l42) (road l3 l24) (road l30 l15) (road l30 l38) (road l31 l45) (road l31 l47) (road l32 l22) (road l33 l2) (road l33 l5) (road l34 l26) (road l35 l37) (road l36 l15) (road l37 l39) (road l38 l6) (road l39 l13) (road l39 l37) (road l4 l45) (road l40 l1) (road l40 l28) (road l40 l41) (road l41 l40) (road l42 l41) (road l43 l24) (road l43 l34) (road l44 l9) (road l45 l31) (road l46 l3) (road l47 l32) (road l5 l33) (road l6 l5) (road l7 l10) (road l8 l36) (road l9 l16) (road l9 l44) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l29) (spare-in l30) (spare-in l33) (spare-in l35) (spare-in l36) (spare-in l38) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l18))
    (:goal (vehicle-at l10))
)