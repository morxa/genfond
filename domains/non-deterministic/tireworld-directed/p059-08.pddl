(define (problem tireworld-059-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l6 l7 l8 l9 - location)
    (:init (road l0 l22) (road l1 l52) (road l10 l40) (road l11 l28) (road l12 l25) (road l12 l5) (road l13 l9) (road l14 l38) (road l14 l4) (road l15 l31) (road l16 l39) (road l17 l16) (road l18 l19) (road l18 l58) (road l19 l35) (road l2 l29) (road l2 l3) (road l20 l37) (road l20 l4) (road l21 l46) (road l21 l48) (road l22 l56) (road l23 l54) (road l24 l6) (road l25 l0) (road l25 l12) (road l26 l38) (road l27 l52) (road l28 l10) (road l29 l2) (road l3 l11) (road l30 l39) (road l30 l57) (road l31 l13) (road l32 l44) (road l32 l45) (road l33 l1) (road l33 l40) (road l34 l35) (road l34 l47) (road l34 l53) (road l35 l34) (road l36 l42) (road l36 l48) (road l37 l43) (road l38 l14) (road l38 l6) (road l39 l16) (road l39 l30) (road l4 l20) (road l40 l33) (road l41 l54) (road l41 l55) (road l42 l36) (road l42 l49) (road l43 l44) (road l44 l32) (road l45 l58) (road l46 l21) (road l46 l26) (road l47 l50) (road l48 l21) (road l48 l36) (road l49 l42) (road l5 l12) (road l50 l51) (road l51 l23) (road l52 l27) (road l52 l8) (road l53 l29) (road l53 l34) (road l54 l41) (road l55 l31) (road l55 l41) (road l56 l22) (road l56 l49) (road l57 l15) (road l57 l30) (road l58 l18) (road l6 l24) (road l6 l38) (road l7 l17) (road l8 l22) (road l8 l27) (road l8 l52) (road l9 l13) (road l9 l5) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l56) (spare-in l57) (spare-in l6) (spare-in l9) (vehicle-at l7))
    (:goal (vehicle-at l24))
)