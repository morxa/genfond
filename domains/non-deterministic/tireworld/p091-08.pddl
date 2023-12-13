(define (problem tireworld-091-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 - location)
    (:init (road l0 l54) (road l0 l58) (road l1 l24) (road l1 l90) (road l10 l49) (road l10 l81) (road l11 l32) (road l11 l34) (road l12 l57) (road l12 l87) (road l13 l65) (road l13 l76) (road l14 l61) (road l14 l88) (road l15 l53) (road l15 l72) (road l16 l17) (road l16 l31) (road l17 l16) (road l17 l8) (road l18 l47) (road l18 l85) (road l19 l2) (road l2 l19) (road l2 l72) (road l20 l40) (road l20 l90) (road l21 l77) (road l21 l80) (road l22 l70) (road l22 l77) (road l23 l56) (road l23 l85) (road l24 l1) (road l24 l36) (road l25 l38) (road l25 l66) (road l26 l4) (road l26 l60) (road l27 l66) (road l27 l75) (road l28 l30) (road l28 l79) (road l29 l59) (road l29 l76) (road l29 l85) (road l3 l64) (road l3 l67) (road l30 l28) (road l30 l38) (road l31 l16) (road l31 l83) (road l32 l11) (road l32 l61) (road l33 l64) (road l33 l65) (road l34 l11) (road l34 l9) (road l35 l48) (road l35 l62) (road l36 l24) (road l36 l82) (road l37 l55) (road l37 l68) (road l38 l25) (road l38 l30) (road l39 l5) (road l39 l51) (road l4 l26) (road l4 l62) (road l40 l20) (road l40 l71) (road l41 l42) (road l41 l58) (road l42 l41) (road l42 l50) (road l43 l86) (road l43 l89) (road l44 l48) (road l44 l53) (road l45 l51) (road l45 l59) (road l45 l71) (road l46 l60) (road l46 l83) (road l47 l18) (road l47 l49) (road l48 l35) (road l48 l44) (road l49 l10) (road l49 l47) (road l5 l39) (road l5 l86) (road l50 l42) (road l50 l67) (road l51 l39) (road l51 l45) (road l52 l7) (road l52 l88) (road l53 l15) (road l53 l44) (road l54 l0) (road l54 l73) (road l55 l37) (road l55 l80) (road l56 l23) (road l56 l82) (road l57 l12) (road l57 l73) (road l58 l0) (road l58 l41) (road l59 l29) (road l59 l45) (road l6 l78) (road l6 l84) (road l60 l26) (road l60 l46) (road l61 l14) (road l61 l32) (road l62 l35) (road l62 l4) (road l63 l68) (road l63 l69) (road l64 l3) (road l64 l33) (road l65 l13) (road l65 l33) (road l66 l25) (road l66 l27) (road l67 l3) (road l67 l50) (road l68 l37) (road l68 l63) (road l69 l63) (road l69 l81) (road l7 l52) (road l7 l89) (road l70 l22) (road l70 l74) (road l71 l40) (road l71 l45) (road l72 l15) (road l72 l2) (road l73 l54) (road l73 l57) (road l74 l70) (road l75 l27) (road l75 l8) (road l76 l13) (road l76 l29) (road l76 l79) (road l77 l21) (road l77 l22) (road l78 l6) (road l78 l87) (road l79 l28) (road l79 l76) (road l8 l17) (road l8 l75) (road l80 l21) (road l80 l55) (road l81 l10) (road l81 l69) (road l82 l36) (road l82 l56) (road l83 l31) (road l83 l46) (road l84 l6) (road l84 l9) (road l85 l18) (road l85 l23) (road l85 l29) (road l86 l43) (road l86 l5) (road l87 l12) (road l87 l78) (road l88 l14) (road l88 l52) (road l89 l43) (road l89 l7) (road l9 l34) (road l9 l84) (road l90 l1) (road l90 l20) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l88) (spare-in l89) (spare-in l9) (spare-in l90) (vehicle-at l19))
    (:goal (vehicle-at l74))
)