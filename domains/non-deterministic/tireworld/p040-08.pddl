(define (problem tireworld-040-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l5) (road l0 l7) (road l1 l22) (road l1 l37) (road l10 l30) (road l10 l39) (road l11 l12) (road l11 l18) (road l12 l11) (road l12 l3) (road l13 l27) (road l13 l5) (road l14 l24) (road l14 l25) (road l15 l28) (road l15 l9) (road l16 l32) (road l16 l38) (road l16 l7) (road l17 l20) (road l18 l11) (road l18 l20) (road l19 l32) (road l19 l36) (road l19 l37) (road l19 l8) (road l2 l35) (road l2 l6) (road l20 l17) (road l20 l18) (road l21 l25) (road l21 l34) (road l22 l1) (road l22 l39) (road l23 l29) (road l23 l37) (road l24 l14) (road l24 l31) (road l24 l33) (road l24 l35) (road l25 l14) (road l25 l21) (road l25 l36) (road l26 l28) (road l26 l30) (road l26 l31) (road l26 l34) (road l27 l13) (road l27 l3) (road l27 l9) (road l28 l15) (road l28 l26) (road l29 l23) (road l29 l4) (road l3 l12) (road l3 l27) (road l30 l10) (road l30 l26) (road l31 l24) (road l31 l26) (road l32 l16) (road l32 l19) (road l33 l24) (road l34 l21) (road l34 l26) (road l35 l2) (road l35 l24) (road l35 l4) (road l36 l19) (road l36 l25) (road l37 l1) (road l37 l19) (road l37 l23) (road l38 l16) (road l38 l6) (road l39 l10) (road l39 l22) (road l4 l29) (road l4 l35) (road l4 l8) (road l5 l0) (road l5 l13) (road l6 l2) (road l6 l38) (road l7 l0) (road l7 l16) (road l8 l19) (road l8 l4) (road l9 l15) (road l9 l27) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l34) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l33))
    (:goal (vehicle-at l17))
)