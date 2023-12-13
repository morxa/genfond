(define (problem tireworld-042-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l13) (road l0 l16) (road l1 l15) (road l1 l6) (road l10 l13) (road l10 l30) (road l11 l29) (road l11 l3) (road l12 l16) (road l12 l21) (road l12 l22) (road l13 l0) (road l13 l10) (road l14 l19) (road l14 l2) (road l15 l1) (road l15 l2) (road l16 l0) (road l16 l12) (road l17 l24) (road l17 l33) (road l18 l39) (road l18 l9) (road l19 l14) (road l19 l20) (road l2 l14) (road l2 l15) (road l2 l21) (road l2 l23) (road l2 l26) (road l20 l19) (road l21 l12) (road l21 l2) (road l22 l12) (road l22 l3) (road l22 l37) (road l23 l2) (road l23 l28) (road l23 l32) (road l24 l17) (road l24 l32) (road l24 l4) (road l25 l35) (road l25 l6) (road l26 l2) (road l26 l30) (road l27 l4) (road l27 l8) (road l28 l23) (road l28 l32) (road l29 l11) (road l29 l31) (road l3 l11) (road l3 l22) (road l30 l10) (road l30 l26) (road l31 l29) (road l31 l39) (road l32 l23) (road l32 l24) (road l32 l28) (road l32 l40) (road l32 l5) (road l33 l17) (road l33 l38) (road l34 l5) (road l34 l7) (road l35 l25) (road l35 l40) (road l35 l5) (road l36 l41) (road l36 l8) (road l37 l22) (road l37 l38) (road l38 l33) (road l38 l37) (road l39 l18) (road l39 l31) (road l4 l24) (road l4 l27) (road l40 l32) (road l40 l35) (road l41 l36) (road l41 l7) (road l5 l32) (road l5 l34) (road l5 l35) (road l6 l1) (road l6 l25) (road l7 l34) (road l7 l41) (road l8 l27) (road l8 l36) (road l9 l18) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l20))
    (:goal (vehicle-at l9))
)