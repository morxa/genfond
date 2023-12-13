(define (problem tireworld-045-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l16) (road l0 l29) (road l0 l34) (road l0 l35) (road l1 l12) (road l1 l38) (road l10 l21) (road l10 l36) (road l11 l29) (road l11 l6) (road l12 l1) (road l12 l39) (road l13 l28) (road l13 l40) (road l14 l24) (road l15 l39) (road l15 l41) (road l16 l0) (road l16 l24) (road l17 l31) (road l17 l32) (road l18 l37) (road l18 l40) (road l19 l22) (road l19 l42) (road l2 l29) (road l2 l35) (road l20 l37) (road l20 l44) (road l21 l10) (road l21 l8) (road l22 l19) (road l22 l4) (road l23 l3) (road l23 l33) (road l24 l14) (road l24 l16) (road l25 l28) (road l25 l41) (road l26 l27) (road l26 l30) (road l27 l26) (road l27 l32) (road l28 l13) (road l28 l25) (road l29 l0) (road l29 l11) (road l29 l2) (road l3 l23) (road l3 l7) (road l30 l26) (road l30 l5) (road l31 l17) (road l31 l42) (road l32 l17) (road l32 l27) (road l33 l23) (road l34 l0) (road l34 l8) (road l35 l0) (road l35 l2) (road l35 l7) (road l36 l10) (road l36 l38) (road l37 l18) (road l37 l20) (road l38 l1) (road l38 l36) (road l39 l12) (road l39 l15) (road l39 l6) (road l4 l22) (road l4 l9) (road l40 l13) (road l40 l18) (road l41 l15) (road l41 l25) (road l42 l19) (road l42 l31) (road l43 l7) (road l43 l9) (road l44 l20) (road l44 l5) (road l5 l30) (road l5 l44) (road l6 l11) (road l6 l39) (road l7 l3) (road l7 l35) (road l7 l43) (road l8 l21) (road l8 l34) (road l9 l4) (road l9 l43) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l33))
    (:goal (vehicle-at l14))
)