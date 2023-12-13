(define (problem tireworld-072-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l8 l9 - location)
    (:init (road l0 l62) (road l0 l63) (road l1 l39) (road l1 l45) (road l10 l17) (road l10 l43) (road l11 l46) (road l11 l60) (road l12 l71) (road l12 l9) (road l13 l27) (road l13 l48) (road l14 l24) (road l14 l3) (road l15 l53) (road l15 l66) (road l16 l25) (road l16 l71) (road l17 l10) (road l17 l33) (road l18 l28) (road l18 l36) (road l18 l5) (road l19 l27) (road l19 l71) (road l2 l5) (road l2 l8) (road l20 l33) (road l20 l34) (road l21 l31) (road l21 l64) (road l22 l4) (road l22 l51) (road l23 l29) (road l23 l49) (road l24 l14) (road l24 l63) (road l25 l16) (road l25 l68) (road l26 l52) (road l26 l69) (road l27 l13) (road l27 l19) (road l28 l18) (road l28 l4) (road l29 l23) (road l29 l35) (road l3 l14) (road l3 l61) (road l30 l57) (road l30 l59) (road l31 l21) (road l31 l36) (road l32 l55) (road l32 l70) (road l33 l17) (road l33 l20) (road l33 l51) (road l34 l20) (road l34 l67) (road l35 l29) (road l35 l40) (road l36 l18) (road l36 l31) (road l36 l47) (road l37 l4) (road l37 l47) (road l37 l9) (road l38 l45) (road l38 l6) (road l39 l1) (road l39 l65) (road l4 l22) (road l4 l28) (road l4 l37) (road l40 l35) (road l40 l57) (road l41 l60) (road l41 l7) (road l42 l58) (road l42 l6) (road l43 l10) (road l43 l65) (road l44 l52) (road l44 l70) (road l45 l1) (road l45 l38) (road l46 l11) (road l46 l56) (road l47 l36) (road l47 l37) (road l48 l13) (road l48 l69) (road l49 l23) (road l49 l68) (road l5 l18) (road l5 l2) (road l50 l53) (road l50 l54) (road l51 l22) (road l51 l33) (road l52 l26) (road l52 l44) (road l53 l15) (road l53 l50) (road l54 l50) (road l54 l58) (road l55 l32) (road l56 l46) (road l56 l66) (road l57 l30) (road l57 l40) (road l58 l42) (road l58 l54) (road l59 l30) (road l59 l61) (road l6 l38) (road l6 l42) (road l60 l11) (road l60 l41) (road l61 l3) (road l61 l59) (road l62 l0) (road l62 l8) (road l63 l0) (road l63 l24) (road l64 l21) (road l64 l67) (road l65 l39) (road l65 l43) (road l66 l15) (road l66 l56) (road l67 l34) (road l67 l64) (road l68 l25) (road l68 l49) (road l69 l26) (road l69 l48) (road l7 l41) (road l70 l32) (road l70 l44) (road l71 l12) (road l71 l16) (road l71 l19) (road l8 l2) (road l8 l62) (road l9 l12) (road l9 l37) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l38) (spare-in l39) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l70) (spare-in l71) (spare-in l8) (vehicle-at l55))
    (:goal (vehicle-at l7))
)