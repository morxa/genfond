(define (problem tireworld-040-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l22) (road l0 l4) (road l0 l8) (road l1 l17) (road l1 l3) (road l10 l14) (road l10 l5) (road l11 l17) (road l11 l37) (road l12 l3) (road l12 l31) (road l13 l26) (road l13 l28) (road l14 l32) (road l15 l29) (road l16 l26) (road l16 l35) (road l17 l1) (road l17 l11) (road l18 l19) (road l18 l28) (road l18 l4) (road l19 l18) (road l19 l23) (road l2 l27) (road l2 l6) (road l20 l33) (road l20 l5) (road l21 l38) (road l21 l39) (road l22 l24) (road l23 l19) (road l23 l31) (road l24 l39) (road l25 l35) (road l26 l13) (road l27 l7) (road l28 l13) (road l28 l18) (road l29 l30) (road l3 l12) (road l30 l29) (road l30 l36) (road l31 l15) (road l31 l23) (road l32 l14) (road l32 l37) (road l33 l20) (road l34 l10) (road l34 l8) (road l35 l16) (road l35 l25) (road l36 l6) (road l37 l11) (road l38 l21) (road l38 l33) (road l38 l9) (road l39 l21) (road l39 l24) (road l4 l0) (road l5 l10) (road l5 l20) (road l6 l2) (road l6 l36) (road l7 l25) (road l8 l0) (road l8 l34) (road l9 l38) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l15))
    (:goal (vehicle-at l9))
)