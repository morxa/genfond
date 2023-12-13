(define (problem tireworld-100-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 l97 l98 l99 - location)
    (:init (road l0 l54) (road l0 l91) (road l1 l14) (road l1 l85) (road l10 l2) (road l10 l75) (road l11 l56) (road l11 l61) (road l12 l33) (road l12 l37) (road l13 l58) (road l13 l59) (road l14 l1) (road l14 l25) (road l15 l28) (road l15 l57) (road l16 l25) (road l16 l34) (road l17 l2) (road l17 l24) (road l18 l60) (road l18 l88) (road l19 l42) (road l19 l5) (road l2 l10) (road l2 l17) (road l20 l23) (road l20 l79) (road l21 l38) (road l21 l89) (road l22 l34) (road l22 l92) (road l23 l20) (road l23 l82) (road l24 l17) (road l24 l39) (road l25 l14) (road l25 l16) (road l26 l32) (road l26 l6) (road l27 l68) (road l27 l8) (road l28 l15) (road l28 l73) (road l28 l84) (road l29 l44) (road l29 l80) (road l3 l63) (road l30 l47) (road l30 l76) (road l31 l71) (road l31 l88) (road l32 l26) (road l32 l42) (road l32 l58) (road l32 l81) (road l33 l12) (road l33 l70) (road l34 l16) (road l34 l22) (road l35 l39) (road l35 l92) (road l36 l94) (road l36 l97) (road l37 l12) (road l37 l74) (road l38 l21) (road l38 l48) (road l39 l24) (road l39 l35) (road l4 l86) (road l4 l96) (road l40 l41) (road l41 l40) (road l41 l94) (road l42 l19) (road l42 l32) (road l42 l46) (road l43 l66) (road l43 l82) (road l44 l29) (road l44 l52) (road l45 l76) (road l45 l84) (road l46 l42) (road l46 l5) (road l46 l83) (road l47 l30) (road l47 l87) (road l48 l38) (road l48 l64) (road l49 l57) (road l49 l95) (road l5 l19) (road l5 l46) (road l50 l53) (road l50 l68) (road l51 l9) (road l51 l99) (road l52 l44) (road l52 l97) (road l53 l50) (road l53 l72) (road l54 l0) (road l54 l73) (road l54 l74) (road l55 l66) (road l55 l71) (road l56 l11) (road l56 l87) (road l57 l15) (road l57 l49) (road l58 l13) (road l58 l32) (road l59 l13) (road l59 l69) (road l59 l99) (road l6 l26) (road l6 l78) (road l60 l18) (road l60 l96) (road l61 l11) (road l61 l85) (road l62 l65) (road l62 l98) (road l63 l3) (road l63 l78) (road l64 l48) (road l64 l86) (road l65 l62) (road l65 l71) (road l66 l43) (road l66 l55) (road l67 l72) (road l67 l81) (road l68 l27) (road l68 l50) (road l69 l59) (road l69 l75) (road l7 l77) (road l7 l79) (road l70 l33) (road l70 l95) (road l71 l31) (road l71 l55) (road l71 l65) (road l72 l53) (road l72 l67) (road l73 l28) (road l73 l54) (road l74 l37) (road l74 l54) (road l75 l10) (road l75 l69) (road l76 l30) (road l76 l45) (road l77 l7) (road l77 l90) (road l78 l6) (road l78 l63) (road l79 l20) (road l79 l7) (road l8 l27) (road l8 l9) (road l80 l29) (road l80 l90) (road l81 l32) (road l81 l67) (road l82 l23) (road l82 l43) (road l83 l46) (road l83 l98) (road l84 l28) (road l84 l45) (road l85 l1) (road l85 l61) (road l86 l4) (road l86 l64) (road l87 l47) (road l87 l56) (road l88 l18) (road l88 l31) (road l89 l21) (road l89 l93) (road l9 l51) (road l9 l8) (road l90 l77) (road l90 l80) (road l91 l0) (road l91 l93) (road l92 l22) (road l92 l35) (road l93 l89) (road l93 l91) (road l94 l36) (road l94 l41) (road l95 l49) (road l95 l70) (road l96 l4) (road l96 l60) (road l97 l36) (road l97 l52) (road l98 l62) (road l98 l83) (road l99 l51) (road l99 l59) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l88) (spare-in l89) (spare-in l9) (spare-in l90) (spare-in l91) (spare-in l92) (spare-in l93) (spare-in l94) (spare-in l95) (spare-in l96) (spare-in l97) (spare-in l98) (spare-in l99) (vehicle-at l3))
    (:goal (vehicle-at l40))
)