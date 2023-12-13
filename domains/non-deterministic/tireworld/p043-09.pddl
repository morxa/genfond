(define (problem tireworld-043-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l19) (road l0 l34) (road l0 l37) (road l1 l23) (road l1 l31) (road l10 l30) (road l10 l32) (road l11 l40) (road l11 l41) (road l12 l22) (road l12 l8) (road l13 l27) (road l13 l32) (road l14 l27) (road l14 l9) (road l15 l34) (road l15 l36) (road l16 l18) (road l16 l29) (road l16 l33) (road l17 l26) (road l17 l7) (road l18 l16) (road l18 l27) (road l19 l0) (road l19 l6) (road l2 l24) (road l2 l39) (road l20 l21) (road l20 l35) (road l21 l20) (road l21 l39) (road l22 l12) (road l22 l42) (road l23 l1) (road l23 l3) (road l24 l2) (road l25 l29) (road l25 l37) (road l26 l17) (road l26 l28) (road l27 l13) (road l27 l14) (road l27 l18) (road l28 l26) (road l28 l30) (road l29 l16) (road l29 l25) (road l3 l23) (road l3 l36) (road l30 l10) (road l30 l28) (road l31 l1) (road l31 l40) (road l32 l10) (road l32 l13) (road l33 l16) (road l33 l4) (road l34 l0) (road l34 l15) (road l35 l20) (road l35 l5) (road l36 l15) (road l36 l3) (road l36 l7) (road l37 l0) (road l37 l25) (road l37 l9) (road l38 l42) (road l38 l5) (road l39 l2) (road l39 l21) (road l4 l33) (road l4 l8) (road l40 l11) (road l40 l31) (road l41 l11) (road l41 l6) (road l42 l22) (road l42 l38) (road l5 l35) (road l5 l38) (road l6 l19) (road l6 l41) (road l7 l17) (road l7 l36) (road l8 l12) (road l8 l4) (road l9 l14) (road l9 l37) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l24))
    (:goal (vehicle-at l30))
)