(define (problem tireworld-063-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l7 l8 l9 - location)
    (:init (road l0 l31) (road l1 l30) (road l10 l6) (road l11 l37) (road l12 l49) (road l13 l17) (road l14 l44) (road l15 l33) (road l16 l42) (road l17 l61) (road l18 l60) (road l19 l5) (road l2 l16) (road l20 l62) (road l21 l28) (road l22 l12) (road l23 l8) (road l24 l51) (road l25 l58) (road l26 l39) (road l27 l50) (road l28 l18) (road l28 l27) (road l29 l26) (road l3 l55) (road l30 l48) (road l31 l60) (road l32 l11) (road l32 l34) (road l33 l13) (road l34 l16) (road l35 l14) (road l36 l23) (road l37 l29) (road l38 l54) (road l39 l45) (road l4 l25) (road l40 l9) (road l41 l20) (road l42 l56) (road l43 l44) (road l44 l53) (road l45 l57) (road l46 l41) (road l47 l10) (road l48 l15) (road l49 l1) (road l5 l2) (road l50 l35) (road l51 l19) (road l52 l0) (road l53 l59) (road l54 l7) (road l55 l43) (road l56 l22) (road l57 l21) (road l57 l46) (road l58 l52) (road l59 l47) (road l6 l36) (road l60 l24) (road l61 l40) (road l62 l4) (road l7 l32) (road l8 l38) (road l9 l3) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l40) (spare-in l41) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l47) (spare-in l49) (spare-in l5) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l8) (spare-in l9) (vehicle-at l2))
    (:goal (vehicle-at l42))
)