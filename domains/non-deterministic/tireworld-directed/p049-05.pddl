(define (problem tireworld-049-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l0 l41) (road l0 l9) (road l1 l0) (road l1 l47) (road l10 l25) (road l10 l48) (road l11 l4) (road l11 l7) (road l12 l27) (road l12 l35) (road l13 l8) (road l14 l45) (road l15 l22) (road l16 l23) (road l17 l20) (road l17 l9) (road l18 l22) (road l18 l37) (road l19 l25) (road l2 l29) (road l20 l17) (road l20 l25) (road l21 l34) (road l22 l15) (road l22 l18) (road l23 l16) (road l23 l8) (road l24 l20) (road l24 l21) (road l25 l10) (road l25 l19) (road l25 l20) (road l25 l41) (road l26 l32) (road l26 l38) (road l26 l39) (road l27 l12) (road l27 l19) (road l28 l4) (road l28 l6) (road l29 l33) (road l3 l37) (road l3 l38) (road l30 l35) (road l31 l40) (road l31 l5) (road l32 l14) (road l32 l26) (road l33 l44) (road l34 l21) (road l34 l36) (road l34 l5) (road l35 l12) (road l35 l13) (road l35 l30) (road l35 l6) (road l36 l34) (road l36 l42) (road l37 l18) (road l37 l3) (road l38 l26) (road l39 l2) (road l4 l11) (road l4 l28) (road l40 l7) (road l41 l0) (road l41 l25) (road l42 l36) (road l42 l46) (road l43 l24) (road l44 l33) (road l44 l46) (road l45 l14) (road l45 l30) (road l46 l42) (road l46 l44) (road l47 l28) (road l48 l15) (road l5 l31) (road l6 l28) (road l6 l35) (road l7 l11) (road l8 l13) (road l8 l23) (road l8 l43) (road l9 l0) (road l9 l17) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l27))
    (:goal (vehicle-at l16))
)