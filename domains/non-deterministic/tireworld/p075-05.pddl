(define (problem tireworld-075-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l8 l9 - location)
    (:init (road l0 l18) (road l0 l74) (road l0 l9) (road l1 l14) (road l1 l42) (road l10 l12) (road l10 l69) (road l11 l13) (road l11 l20) (road l12 l10) (road l12 l2) (road l13 l11) (road l14 l1) (road l14 l68) (road l15 l27) (road l15 l31) (road l16 l17) (road l16 l62) (road l17 l16) (road l17 l3) (road l18 l0) (road l18 l29) (road l19 l48) (road l19 l5) (road l2 l12) (road l2 l40) (road l20 l11) (road l20 l28) (road l21 l30) (road l21 l45) (road l22 l44) (road l22 l64) (road l23 l59) (road l23 l63) (road l24 l28) (road l24 l49) (road l25 l44) (road l25 l6) (road l26 l47) (road l26 l52) (road l27 l15) (road l27 l50) (road l28 l20) (road l28 l24) (road l29 l18) (road l29 l34) (road l29 l37) (road l29 l6) (road l3 l17) (road l3 l39) (road l30 l21) (road l30 l37) (road l30 l38) (road l30 l73) (road l30 l74) (road l31 l15) (road l31 l53) (road l32 l5) (road l32 l8) (road l33 l39) (road l33 l72) (road l34 l29) (road l34 l67) (road l35 l68) (road l35 l71) (road l36 l58) (road l36 l7) (road l37 l29) (road l37 l30) (road l38 l30) (road l38 l9) (road l39 l3) (road l39 l33) (road l4 l49) (road l4 l56) (road l40 l2) (road l40 l53) (road l41 l62) (road l41 l67) (road l42 l1) (road l43 l51) (road l43 l69) (road l44 l22) (road l44 l25) (road l45 l21) (road l45 l47) (road l46 l60) (road l46 l7) (road l47 l26) (road l47 l45) (road l48 l19) (road l48 l72) (road l49 l24) (road l49 l4) (road l5 l19) (road l5 l32) (road l50 l27) (road l50 l73) (road l51 l43) (road l51 l57) (road l52 l26) (road l52 l60) (road l53 l31) (road l53 l40) (road l54 l59) (road l54 l64) (road l55 l63) (road l55 l8) (road l56 l4) (road l56 l58) (road l57 l51) (road l57 l70) (road l58 l36) (road l58 l56) (road l59 l23) (road l59 l54) (road l6 l25) (road l6 l29) (road l60 l46) (road l60 l52) (road l60 l67) (road l61 l65) (road l61 l66) (road l62 l16) (road l62 l41) (road l63 l23) (road l63 l55) (road l64 l22) (road l64 l54) (road l65 l61) (road l65 l71) (road l66 l61) (road l66 l70) (road l67 l34) (road l67 l41) (road l67 l60) (road l68 l14) (road l68 l35) (road l69 l10) (road l69 l43) (road l7 l36) (road l7 l46) (road l70 l57) (road l70 l66) (road l71 l35) (road l71 l65) (road l72 l33) (road l72 l48) (road l73 l30) (road l73 l50) (road l74 l0) (road l74 l30) (road l8 l32) (road l8 l55) (road l9 l0) (road l9 l38) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l30) (spare-in l31) (spare-in l35) (spare-in l36) (spare-in l4) (spare-in l40) (spare-in l43) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l60) (spare-in l61) (spare-in l65) (spare-in l66) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l73) (vehicle-at l42))
    (:goal (vehicle-at l13))
)