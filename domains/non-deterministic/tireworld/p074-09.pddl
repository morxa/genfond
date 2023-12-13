(define (problem tireworld-074-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l8 l9 - location)
    (:init (road l0 l23) (road l0 l9) (road l1 l19) (road l1 l32) (road l10 l26) (road l10 l34) (road l10 l50) (road l11 l70) (road l11 l9) (road l12 l40) (road l12 l58) (road l13 l38) (road l13 l56) (road l14 l57) (road l14 l66) (road l15 l2) (road l15 l68) (road l16 l2) (road l16 l4) (road l17 l52) (road l17 l53) (road l18 l50) (road l18 l54) (road l19 l1) (road l19 l61) (road l2 l15) (road l2 l16) (road l20 l48) (road l20 l52) (road l21 l30) (road l21 l53) (road l22 l48) (road l22 l67) (road l23 l0) (road l23 l29) (road l24 l27) (road l24 l40) (road l25 l62) (road l25 l64) (road l26 l10) (road l26 l58) (road l27 l24) (road l27 l70) (road l28 l6) (road l28 l69) (road l29 l23) (road l29 l7) (road l3 l42) (road l3 l63) (road l30 l21) (road l30 l35) (road l31 l41) (road l32 l1) (road l32 l69) (road l33 l4) (road l33 l63) (road l34 l10) (road l34 l68) (road l35 l30) (road l35 l5) (road l36 l42) (road l36 l46) (road l37 l60) (road l37 l8) (road l38 l13) (road l38 l39) (road l39 l38) (road l39 l49) (road l4 l16) (road l4 l33) (road l4 l46) (road l4 l59) (road l40 l12) (road l40 l24) (road l41 l31) (road l41 l54) (road l42 l3) (road l42 l36) (road l42 l72) (road l43 l45) (road l43 l51) (road l44 l55) (road l44 l7) (road l45 l43) (road l45 l6) (road l46 l36) (road l46 l4) (road l46 l62) (road l47 l62) (road l47 l68) (road l48 l20) (road l48 l22) (road l49 l39) (road l49 l71) (road l5 l35) (road l5 l66) (road l50 l10) (road l50 l18) (road l51 l43) (road l51 l8) (road l52 l17) (road l52 l20) (road l53 l17) (road l53 l21) (road l54 l18) (road l54 l41) (road l55 l44) (road l55 l62) (road l56 l13) (road l56 l73) (road l57 l14) (road l57 l64) (road l58 l12) (road l58 l26) (road l59 l4) (road l59 l65) (road l59 l71) (road l6 l28) (road l6 l45) (road l60 l37) (road l60 l72) (road l61 l19) (road l62 l25) (road l62 l46) (road l62 l47) (road l62 l55) (road l63 l3) (road l63 l33) (road l63 l65) (road l64 l25) (road l64 l57) (road l65 l59) (road l65 l63) (road l66 l14) (road l66 l5) (road l67 l22) (road l67 l73) (road l68 l15) (road l68 l34) (road l68 l47) (road l69 l28) (road l69 l32) (road l7 l29) (road l7 l44) (road l70 l11) (road l70 l27) (road l71 l49) (road l71 l59) (road l72 l42) (road l72 l60) (road l73 l56) (road l73 l67) (road l8 l37) (road l8 l51) (road l9 l0) (road l9 l11) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l8) (spare-in l9) (vehicle-at l31))
    (:goal (vehicle-at l61))
)