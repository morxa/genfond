(define (problem tireworld-059-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l6 l7 l8 l9 - location)
    (:init (road l0 l50) (road l1 l25) (road l10 l4) (road l11 l31) (road l11 l46) (road l13 l37) (road l13 l41) (road l14 l37) (road l15 l0) (road l16 l10) (road l17 l43) (road l18 l42) (road l18 l57) (road l19 l21) (road l19 l33) (road l2 l20) (road l2 l44) (road l20 l45) (road l21 l23) (road l21 l40) (road l22 l56) (road l23 l7) (road l24 l49) (road l25 l32) (road l26 l39) (road l27 l1) (road l27 l53) (road l28 l22) (road l28 l54) (road l29 l9) (road l3 l36) (road l30 l16) (road l30 l43) (road l31 l58) (road l32 l26) (road l33 l19) (road l34 l3) (road l35 l41) (road l35 l47) (road l36 l48) (road l37 l13) (road l37 l50) (road l38 l53) (road l39 l34) (road l4 l19) (road l40 l12) (road l41 l15) (road l41 l35) (road l42 l55) (road l43 l30) (road l44 l2) (road l45 l33) (road l46 l11) (road l47 l35) (road l47 l44) (road l48 l14) (road l49 l24) (road l49 l46) (road l5 l17) (road l50 l53) (road l51 l6) (road l52 l57) (road l53 l27) (road l53 l38) (road l54 l28) (road l54 l8) (road l55 l51) (road l56 l5) (road l57 l18) (road l58 l8) (road l6 l29) (road l6 l38) (road l7 l52) (road l8 l54) (road l8 l58) (road l9 l24) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l21) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l34) (spare-in l35) (spare-in l37) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l46) (spare-in l48) (spare-in l5) (spare-in l52) (spare-in l53) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l43))
    (:goal (vehicle-at l12))
)