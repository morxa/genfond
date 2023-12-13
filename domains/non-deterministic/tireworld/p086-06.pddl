(define (problem tireworld-086-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l9 - location)
    (:init (road l0 l24) (road l0 l47) (road l1 l4) (road l1 l9) (road l10 l50) (road l10 l67) (road l11 l13) (road l11 l24) (road l11 l34) (road l12 l52) (road l12 l82) (road l13 l11) (road l13 l84) (road l14 l40) (road l14 l71) (road l15 l55) (road l15 l65) (road l16 l17) (road l16 l77) (road l17 l16) (road l17 l29) (road l18 l19) (road l18 l44) (road l19 l18) (road l19 l38) (road l2 l63) (road l2 l84) (road l20 l32) (road l20 l43) (road l21 l22) (road l21 l76) (road l22 l21) (road l22 l48) (road l23 l62) (road l23 l83) (road l24 l0) (road l24 l11) (road l25 l61) (road l25 l65) (road l26 l32) (road l26 l78) (road l27 l39) (road l27 l44) (road l28 l55) (road l28 l80) (road l29 l17) (road l29 l46) (road l3 l33) (road l3 l42) (road l30 l5) (road l30 l74) (road l31 l5) (road l31 l79) (road l32 l20) (road l32 l26) (road l33 l3) (road l33 l7) (road l34 l11) (road l34 l49) (road l35 l73) (road l35 l78) (road l36 l56) (road l36 l85) (road l37 l38) (road l37 l68) (road l38 l19) (road l38 l37) (road l38 l48) (road l39 l27) (road l39 l53) (road l4 l1) (road l4 l79) (road l40 l14) (road l40 l51) (road l41 l80) (road l42 l3) (road l42 l56) (road l43 l20) (road l43 l62) (road l44 l18) (road l44 l27) (road l44 l6) (road l44 l85) (road l45 l57) (road l46 l29) (road l46 l7) (road l46 l9) (road l47 l0) (road l47 l60) (road l48 l22) (road l48 l38) (road l49 l34) (road l49 l85) (road l5 l30) (road l5 l31) (road l50 l10) (road l50 l81) (road l51 l40) (road l51 l7) (road l52 l12) (road l52 l73) (road l53 l39) (road l53 l70) (road l54 l66) (road l54 l69) (road l55 l15) (road l55 l28) (road l56 l36) (road l56 l42) (road l57 l45) (road l57 l66) (road l58 l61) (road l58 l71) (road l59 l64) (road l59 l71) (road l6 l44) (road l6 l71) (road l60 l47) (road l60 l83) (road l61 l25) (road l61 l58) (road l62 l23) (road l62 l43) (road l63 l2) (road l63 l81) (road l64 l59) (road l64 l82) (road l65 l15) (road l65 l25) (road l65 l72) (road l66 l54) (road l66 l57) (road l67 l10) (road l67 l76) (road l68 l37) (road l68 l69) (road l69 l54) (road l69 l68) (road l7 l33) (road l7 l46) (road l7 l51) (road l70 l53) (road l70 l75) (road l71 l14) (road l71 l58) (road l71 l59) (road l71 l6) (road l71 l74) (road l71 l75) (road l72 l65) (road l72 l8) (road l73 l35) (road l73 l52) (road l74 l30) (road l74 l71) (road l75 l70) (road l75 l71) (road l76 l21) (road l76 l67) (road l77 l16) (road l77 l8) (road l78 l26) (road l78 l35) (road l79 l31) (road l79 l4) (road l8 l72) (road l8 l77) (road l80 l28) (road l80 l41) (road l81 l50) (road l81 l63) (road l82 l12) (road l82 l64) (road l83 l23) (road l83 l60) (road l84 l13) (road l84 l2) (road l85 l36) (road l85 l44) (road l85 l49) (road l9 l1) (road l9 l46) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l43) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l55) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l60) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l9) (vehicle-at l45))
    (:goal (vehicle-at l41))
)