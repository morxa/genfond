(define (problem tireworld-044-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l18) (road l0 l41) (road l1 l26) (road l10 l18) (road l11 l31) (road l11 l37) (road l12 l31) (road l12 l7) (road l13 l17) (road l14 l27) (road l14 l39) (road l15 l34) (road l15 l36) (road l15 l41) (road l16 l32) (road l16 l7) (road l17 l13) (road l17 l22) (road l18 l0) (road l18 l10) (road l19 l18) (road l2 l31) (road l2 l6) (road l20 l33) (road l21 l32) (road l22 l17) (road l22 l9) (road l23 l4) (road l25 l24) (road l25 l40) (road l25 l5) (road l26 l1) (road l26 l3) (road l27 l14) (road l27 l33) (road l28 l15) (road l28 l32) (road l29 l20) (road l3 l10) (road l3 l26) (road l30 l1) (road l31 l11) (road l31 l12) (road l32 l16) (road l32 l28) (road l33 l20) (road l33 l27) (road l34 l13) (road l34 l15) (road l35 l43) (road l36 l15) (road l36 l40) (road l37 l38) (road l37 l42) (road l37 l8) (road l38 l37) (road l38 l5) (road l38 l6) (road l39 l14) (road l39 l30) (road l4 l21) (road l4 l23) (road l40 l25) (road l40 l36) (road l41 l0) (road l41 l15) (road l41 l42) (road l42 l37) (road l43 l15) (road l43 l29) (road l43 l35) (road l5 l25) (road l6 l2) (road l6 l38) (road l7 l16) (road l7 l35) (road l8 l23) (road l9 l20) (road l9 l22) (spare-in l0) (spare-in l13) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l20) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l33) (spare-in l35) (spare-in l36) (spare-in l38) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l5) (spare-in l6) (spare-in l8) (spare-in l9) (vehicle-at l19))
    (:goal (vehicle-at l24))
)