(define (problem tireworld-077-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l8 l9 - location)
    (:init (road l0 l41) (road l0 l46) (road l1 l37) (road l1 l70) (road l10 l28) (road l10 l64) (road l11 l67) (road l12 l61) (road l12 l67) (road l13 l17) (road l13 l59) (road l14 l40) (road l14 l65) (road l15 l42) (road l15 l45) (road l16 l43) (road l16 l60) (road l17 l13) (road l17 l72) (road l18 l20) (road l18 l52) (road l19 l28) (road l2 l30) (road l2 l4) (road l20 l18) (road l20 l44) (road l20 l74) (road l21 l47) (road l21 l64) (road l22 l25) (road l22 l74) (road l23 l11) (road l23 l50) (road l24 l52) (road l24 l73) (road l25 l22) (road l25 l73) (road l26 l29) (road l26 l72) (road l27 l33) (road l27 l9) (road l28 l10) (road l29 l26) (road l29 l57) (road l3 l44) (road l3 l58) (road l30 l2) (road l30 l69) (road l31 l6) (road l31 l65) (road l32 l36) (road l32 l38) (road l33 l27) (road l33 l5) (road l33 l51) (road l33 l8) (road l34 l49) (road l34 l7) (road l35 l39) (road l35 l75) (road l36 l32) (road l36 l55) (road l37 l1) (road l37 l66) (road l38 l32) (road l38 l48) (road l39 l35) (road l4 l2) (road l4 l71) (road l40 l14) (road l40 l55) (road l41 l0) (road l41 l69) (road l42 l15) (road l42 l54) (road l43 l16) (road l43 l9) (road l44 l20) (road l44 l3) (road l45 l15) (road l45 l6) (road l46 l0) (road l46 l57) (road l47 l21) (road l47 l5) (road l48 l38) (road l48 l53) (road l49 l19) (road l49 l34) (road l5 l33) (road l5 l47) (road l50 l23) (road l50 l54) (road l51 l33) (road l51 l73) (road l52 l18) (road l52 l24) (road l53 l48) (road l53 l60) (road l54 l42) (road l54 l50) (road l55 l36) (road l55 l40) (road l56 l62) (road l57 l29) (road l57 l46) (road l58 l3) (road l58 l62) (road l59 l13) (road l59 l68) (road l6 l31) (road l6 l45) (road l60 l16) (road l60 l53) (road l60 l73) (road l61 l12) (road l61 l63) (road l62 l56) (road l62 l58) (road l63 l61) (road l63 l66) (road l64 l10) (road l64 l21) (road l65 l14) (road l65 l31) (road l66 l37) (road l66 l63) (road l67 l12) (road l68 l56) (road l68 l59) (road l69 l30) (road l69 l41) (road l7 l34) (road l7 l75) (road l70 l1) (road l71 l4) (road l71 l76) (road l72 l17) (road l72 l26) (road l73 l24) (road l73 l25) (road l73 l51) (road l73 l60) (road l74 l20) (road l74 l22) (road l75 l35) (road l75 l7) (road l76 l71) (road l76 l8) (road l8 l33) (road l8 l76) (road l9 l27) (road l9 l43) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l8) (spare-in l9) (vehicle-at l39))
    (:goal (vehicle-at l70))
)