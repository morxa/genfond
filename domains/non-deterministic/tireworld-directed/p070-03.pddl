(define (problem tireworld-070-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l8 l9 - location)
    (:init (road l0 l36) (road l0 l41) (road l1 l29) (road l10 l24) (road l10 l51) (road l11 l2) (road l11 l68) (road l12 l26) (road l13 l20) (road l14 l23) (road l15 l50) (road l16 l49) (road l17 l52) (road l18 l19) (road l2 l58) (road l20 l26) (road l21 l55) (road l22 l30) (road l23 l14) (road l23 l32) (road l24 l12) (road l25 l40) (road l26 l1) (road l27 l34) (road l28 l33) (road l29 l64) (road l3 l30) (road l3 l37) (road l30 l3) (road l31 l66) (road l32 l66) (road l33 l27) (road l33 l28) (road l34 l38) (road l34 l51) (road l35 l63) (road l36 l0) (road l37 l67) (road l38 l18) (road l39 l5) (road l39 l56) (road l4 l57) (road l4 l60) (road l40 l21) (road l41 l0) (road l41 l17) (road l42 l11) (road l43 l44) (road l43 l5) (road l43 l69) (road l44 l43) (road l44 l47) (road l45 l35) (road l46 l6) (road l47 l44) (road l48 l52) (road l49 l36) (road l5 l15) (road l5 l39) (road l50 l13) (road l50 l15) (road l51 l10) (road l51 l61) (road l52 l7) (road l53 l8) (road l54 l25) (road l55 l65) (road l56 l14) (road l56 l39) (road l57 l4) (road l57 l62) (road l58 l2) (road l58 l48) (road l59 l9) (road l6 l22) (road l60 l4) (road l61 l44) (road l61 l51) (road l62 l54) (road l63 l35) (road l63 l60) (road l64 l29) (road l64 l45) (road l65 l28) (road l65 l55) (road l66 l46) (road l67 l24) (road l68 l16) (road l69 l42) (road l7 l52) (road l7 l59) (road l8 l47) (road l8 l53) (road l9 l53) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l18) (spare-in l21) (spare-in l22) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l40) (spare-in l45) (spare-in l46) (spare-in l54) (spare-in l55) (spare-in l57) (spare-in l6) (spare-in l60) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l7) (vehicle-at l31))
    (:goal (vehicle-at l19))
)