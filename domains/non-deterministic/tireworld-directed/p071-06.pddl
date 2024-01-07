(define (problem tireworld-071-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l8 l9 - location)
    (:init (road l0 l24) (road l0 l3) (road l1 l14) (road l1 l64) (road l10 l12) (road l10 l66) (road l11 l38) (road l11 l46) (road l12 l10) (road l12 l58) (road l13 l19) (road l13 l46) (road l14 l1) (road l14 l32) (road l15 l44) (road l15 l52) (road l16 l17) (road l16 l63) (road l17 l16) (road l17 l53) (road l18 l30) (road l18 l40) (road l19 l13) (road l2 l6) (road l2 l61) (road l20 l55) (road l20 l7) (road l21 l39) (road l21 l41) (road l22 l24) (road l22 l36) (road l23 l57) (road l23 l65) (road l24 l0) (road l24 l22) (road l25 l56) (road l25 l66) (road l26 l31) (road l26 l67) (road l27 l29) (road l27 l45) (road l28 l37) (road l28 l51) (road l29 l27) (road l29 l4) (road l29 l40) (road l3 l0) (road l3 l52) (road l30 l9) (road l31 l26) (road l31 l68) (road l32 l14) (road l32 l36) (road l33 l59) (road l33 l70) (road l34 l45) (road l34 l53) (road l35 l44) (road l35 l5) (road l36 l22) (road l36 l32) (road l37 l28) (road l37 l49) (road l38 l11) (road l38 l68) (road l39 l21) (road l39 l4) (road l4 l29) (road l4 l39) (road l40 l18) (road l40 l29) (road l41 l21) (road l41 l66) (road l42 l62) (road l42 l65) (road l43 l59) (road l43 l8) (road l44 l15) (road l44 l35) (road l45 l27) (road l45 l34) (road l46 l11) (road l46 l13) (road l47 l62) (road l48 l55) (road l48 l61) (road l49 l37) (road l49 l64) (road l5 l35) (road l5 l57) (road l50 l67) (road l50 l69) (road l51 l28) (road l51 l9) (road l52 l15) (road l52 l3) (road l53 l17) (road l53 l34) (road l54 l56) (road l54 l63) (road l55 l20) (road l55 l48) (road l56 l25) (road l56 l54) (road l57 l23) (road l57 l5) (road l58 l12) (road l58 l7) (road l59 l33) (road l59 l43) (road l6 l2) (road l6 l60) (road l60 l6) (road l60 l70) (road l61 l2) (road l61 l48) (road l62 l42) (road l62 l47) (road l63 l16) (road l63 l54) (road l63 l65) (road l64 l1) (road l64 l49) (road l65 l23) (road l65 l42) (road l65 l63) (road l66 l10) (road l66 l25) (road l66 l41) (road l67 l26) (road l67 l50) (road l68 l31) (road l68 l38) (road l69 l50) (road l69 l8) (road l7 l20) (road l7 l58) (road l70 l33) (road l70 l60) (road l8 l43) (road l8 l69) (road l9 l30) (road l9 l51) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l36) (spare-in l38) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l46) (spare-in l48) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l8) (spare-in l9) (vehicle-at l47))
    (:goal (vehicle-at l19))
)