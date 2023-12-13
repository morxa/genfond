(define (problem tireworld-045-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l29) (road l0 l31) (road l0 l5) (road l1 l34) (road l1 l36) (road l10 l22) (road l10 l27) (road l11 l19) (road l11 l35) (road l12 l25) (road l12 l44) (road l13 l34) (road l13 l8) (road l14 l4) (road l14 l5) (road l15 l21) (road l15 l28) (road l16 l41) (road l16 l8) (road l17 l20) (road l17 l37) (road l18 l26) (road l18 l9) (road l19 l11) (road l19 l38) (road l2 l42) (road l2 l43) (road l20 l17) (road l20 l40) (road l21 l15) (road l21 l27) (road l22 l10) (road l22 l35) (road l23 l37) (road l23 l7) (road l24 l4) (road l24 l41) (road l25 l12) (road l25 l26) (road l26 l18) (road l26 l25) (road l27 l10) (road l27 l21) (road l28 l15) (road l28 l39) (road l29 l0) (road l29 l6) (road l3 l39) (road l3 l43) (road l30 l32) (road l30 l33) (road l31 l0) (road l31 l40) (road l32 l30) (road l32 l44) (road l33 l30) (road l33 l36) (road l34 l1) (road l34 l13) (road l35 l11) (road l35 l22) (road l35 l9) (road l36 l1) (road l36 l33) (road l37 l17) (road l37 l23) (road l38 l19) (road l38 l6) (road l39 l28) (road l39 l3) (road l4 l14) (road l4 l24) (road l40 l20) (road l40 l31) (road l41 l16) (road l41 l24) (road l42 l2) (road l43 l2) (road l43 l3) (road l44 l12) (road l44 l32) (road l5 l0) (road l5 l14) (road l6 l29) (road l6 l38) (road l7 l23) (road l8 l13) (road l8 l16) (road l9 l18) (road l9 l35) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l43) (spare-in l44) (spare-in l5) (spare-in l8) (spare-in l9) (vehicle-at l42))
    (:goal (vehicle-at l7))
)