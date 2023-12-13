(define (problem tireworld-070-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l8 l9 - location)
    (:init (road l0 l36) (road l0 l7) (road l1 l16) (road l1 l37) (road l10 l50) (road l10 l68) (road l11 l21) (road l11 l47) (road l12 l47) (road l12 l52) (road l12 l67) (road l13 l15) (road l13 l67) (road l14 l55) (road l14 l56) (road l14 l64) (road l14 l68) (road l15 l13) (road l15 l7) (road l16 l1) (road l16 l57) (road l17 l53) (road l17 l61) (road l18 l63) (road l18 l64) (road l19 l29) (road l19 l59) (road l2 l38) (road l2 l61) (road l20 l44) (road l20 l57) (road l21 l11) (road l21 l62) (road l22 l25) (road l22 l8) (road l23 l40) (road l23 l42) (road l24 l37) (road l24 l63) (road l25 l22) (road l25 l42) (road l26 l4) (road l26 l65) (road l27 l52) (road l27 l8) (road l28 l29) (road l28 l5) (road l29 l19) (road l29 l28) (road l3 l48) (road l3 l9) (road l30 l38) (road l30 l69) (road l31 l58) (road l31 l9) (road l32 l41) (road l32 l65) (road l33 l58) (road l33 l69) (road l34 l4) (road l34 l45) (road l35 l62) (road l35 l66) (road l36 l0) (road l36 l46) (road l37 l1) (road l37 l24) (road l38 l2) (road l38 l30) (road l39 l49) (road l39 l61) (road l4 l26) (road l4 l34) (road l40 l23) (road l40 l51) (road l41 l32) (road l41 l51) (road l42 l23) (road l42 l25) (road l43 l49) (road l43 l60) (road l44 l20) (road l44 l59) (road l45 l34) (road l45 l48) (road l46 l36) (road l46 l54) (road l47 l11) (road l47 l12) (road l48 l3) (road l48 l45) (road l49 l39) (road l49 l43) (road l5 l28) (road l5 l60) (road l50 l10) (road l50 l6) (road l51 l40) (road l51 l41) (road l52 l12) (road l52 l27) (road l53 l17) (road l53 l54) (road l53 l56) (road l54 l46) (road l54 l53) (road l55 l14) (road l55 l61) (road l56 l14) (road l56 l53) (road l57 l16) (road l57 l20) (road l58 l31) (road l58 l33) (road l59 l19) (road l59 l44) (road l6 l50) (road l6 l66) (road l60 l43) (road l60 l5) (road l61 l17) (road l61 l2) (road l61 l39) (road l61 l55) (road l62 l21) (road l62 l35) (road l63 l18) (road l63 l24) (road l64 l14) (road l64 l18) (road l65 l26) (road l65 l32) (road l66 l35) (road l66 l6) (road l67 l12) (road l67 l13) (road l68 l10) (road l68 l14) (road l69 l30) (road l69 l33) (road l7 l0) (road l7 l15) (road l8 l22) (road l8 l27) (road l9 l3) (road l9 l31) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l61))
    (:goal (vehicle-at l56))
)