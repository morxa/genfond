(define (problem tireworld-044-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l22) (road l0 l31) (road l1 l15) (road l1 l40) (road l10 l28) (road l10 l34) (road l10 l43) (road l11 l13) (road l11 l23) (road l12 l27) (road l12 l29) (road l13 l11) (road l13 l21) (road l14 l21) (road l14 l26) (road l15 l1) (road l15 l34) (road l16 l37) (road l16 l9) (road l17 l23) (road l17 l39) (road l18 l26) (road l19 l31) (road l19 l41) (road l2 l20) (road l2 l30) (road l2 l4) (road l20 l2) (road l20 l39) (road l21 l13) (road l21 l14) (road l21 l32) (road l22 l0) (road l22 l30) (road l23 l11) (road l23 l17) (road l24 l28) (road l24 l4) (road l25 l29) (road l25 l8) (road l26 l14) (road l26 l18) (road l27 l12) (road l27 l3) (road l28 l10) (road l28 l24) (road l29 l12) (road l29 l25) (road l3 l27) (road l3 l37) (road l30 l2) (road l30 l22) (road l31 l0) (road l31 l19) (road l32 l21) (road l32 l38) (road l33 l35) (road l33 l39) (road l33 l40) (road l33 l6) (road l34 l10) (road l34 l15) (road l35 l33) (road l35 l7) (road l36 l42) (road l36 l7) (road l37 l16) (road l37 l3) (road l38 l32) (road l38 l41) (road l38 l6) (road l39 l17) (road l39 l20) (road l39 l33) (road l4 l2) (road l4 l24) (road l40 l1) (road l40 l33) (road l41 l19) (road l41 l38) (road l42 l36) (road l42 l8) (road l43 l10) (road l43 l5) (road l5 l43) (road l5 l9) (road l6 l33) (road l6 l38) (road l7 l35) (road l7 l36) (road l8 l25) (road l8 l42) (road l9 l16) (road l9 l5) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l4) (spare-in l41) (spare-in l43) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l38))
    (:goal (vehicle-at l18))
)