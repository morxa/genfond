(define (problem tireworld-080-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l9 - location)
    (:init (road l0 l24) (road l0 l31) (road l1 l54) (road l1 l60) (road l10 l18) (road l10 l48) (road l10 l65) (road l11 l68) (road l11 l77) (road l12 l7) (road l12 l78) (road l13 l22) (road l13 l46) (road l14 l44) (road l14 l56) (road l15 l5) (road l15 l55) (road l16 l42) (road l16 l55) (road l17 l41) (road l17 l67) (road l18 l10) (road l18 l26) (road l19 l4) (road l19 l48) (road l2 l45) (road l2 l57) (road l20 l62) (road l20 l79) (road l21 l49) (road l21 l52) (road l22 l13) (road l22 l66) (road l23 l58) (road l23 l75) (road l24 l0) (road l24 l59) (road l25 l38) (road l25 l61) (road l26 l18) (road l26 l70) (road l27 l30) (road l27 l74) (road l28 l40) (road l28 l60) (road l29 l46) (road l29 l51) (road l3 l56) (road l3 l59) (road l30 l27) (road l30 l40) (road l31 l0) (road l31 l73) (road l32 l33) (road l32 l74) (road l33 l32) (road l33 l47) (road l34 l35) (road l34 l53) (road l35 l34) (road l35 l68) (road l36 l63) (road l36 l67) (road l37 l65) (road l38 l25) (road l38 l5) (road l39 l43) (road l39 l77) (road l4 l19) (road l4 l71) (road l40 l28) (road l40 l30) (road l41 l17) (road l41 l9) (road l42 l16) (road l42 l9) (road l43 l39) (road l43 l64) (road l44 l14) (road l44 l76) (road l45 l2) (road l45 l70) (road l46 l13) (road l46 l29) (road l46 l8) (road l47 l33) (road l47 l72) (road l48 l10) (road l48 l19) (road l49 l21) (road l49 l75) (road l5 l15) (road l5 l38) (road l50 l63) (road l50 l71) (road l51 l29) (road l51 l76) (road l52 l21) (road l52 l69) (road l53 l34) (road l53 l73) (road l54 l1) (road l55 l15) (road l55 l16) (road l56 l14) (road l56 l3) (road l57 l2) (road l57 l66) (road l58 l23) (road l58 l68) (road l59 l24) (road l59 l3) (road l6 l72) (road l6 l78) (road l60 l1) (road l60 l28) (road l61 l25) (road l61 l7) (road l62 l20) (road l62 l8) (road l63 l36) (road l63 l50) (road l64 l43) (road l64 l65) (road l65 l10) (road l65 l37) (road l65 l64) (road l66 l22) (road l66 l57) (road l67 l17) (road l67 l36) (road l68 l11) (road l68 l35) (road l68 l58) (road l69 l52) (road l69 l79) (road l7 l12) (road l7 l61) (road l70 l26) (road l70 l45) (road l71 l4) (road l71 l50) (road l72 l47) (road l72 l6) (road l73 l31) (road l73 l53) (road l74 l27) (road l74 l32) (road l75 l23) (road l75 l49) (road l76 l44) (road l76 l51) (road l77 l11) (road l77 l39) (road l78 l12) (road l78 l6) (road l79 l20) (road l79 l69) (road l8 l46) (road l8 l62) (road l9 l41) (road l9 l42) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l25) (spare-in l27) (spare-in l28) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l36) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l55) (spare-in l58) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l63) (spare-in l65) (spare-in l67) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l74) (spare-in l77) (spare-in l78) (spare-in l9) (vehicle-at l37))
    (:goal (vehicle-at l54))
)